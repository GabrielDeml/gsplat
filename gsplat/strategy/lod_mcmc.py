# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hierarchical MCMC densification strategy.

Pairs with ``gsplat.lod.HierarchicalMcmcDensifier`` and
``gsplat.lod.OutOfCoreAdam`` to drive the LoD training loop. Unlike the
regular ``MCMCStrategy`` which manipulates a persistent on-GPU ParameterDict,
this strategy only sees the per-iteration *active* subset; the persistent
state lives in the densifier / CPU store.

Lifecycle per iteration (driven by the trainer):

    strategy.step_pre_backward(active, step, info)
    render + loss.backward()
    strategy.step_post_backward(active, step, info)
    oc_optim.step(active_indices, active)

The strategy's main jobs are:
    1. Accumulate max screen-space gradient stats from ``info['means2d']``
       into the densifier's running stats.
    2. Fire densification every ``cfg.densify_every`` iterations within
       ``[densify_start, densify_stop]``.
    3. Trigger MCMC-style noise injection on active means.
    4. Trigger cache flushes per ``cfg.cache_full_flush_every``.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch
from torch import Tensor

from .base import Strategy
from .ops import inject_noise_to_position
from ..lod.cache import GpuSptCache
from ..lod.config import LoDConfig
from ..lod.densify import HierarchicalMcmcDensifier
from ..lod.hspt import HSPT


@dataclass
class LoDMCMCStrategy(Strategy):
    """Strategy that drives the out-of-core LoD MCMC training pipeline.

    The strategy does not own the hierarchy / store / cache — the trainer
    constructs those and passes them through ``state`` to each callback.

    Args:
        cfg: LoDConfig. Defaults follow the paper.
    """

    cfg: LoDConfig = field(default_factory=LoDConfig)

    # --- Sanity ---------------------------------------------------------

    def check_sanity(self, params, optimizers) -> None:
        expected = {"means", "scales", "quats", "opacities", "sh0", "shN"}
        for k in expected:
            assert k in params, f"LoDMCMCStrategy: missing param {k!r}"

    # --- Per-iteration hooks -------------------------------------------

    def step_pre_backward(
        self,
        params: Dict[str, Tensor],
        optimizers: Any,
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ) -> None:
        """Retain graph state and prep the info dict.

        If ``info["means2d"]`` is a ``torch.Tensor`` with ``requires_grad``,
        we also call ``retain_grad()`` so we can read its gradient in
        ``step_post_backward``.
        """
        m2d = info.get("means2d")
        if isinstance(m2d, torch.Tensor) and m2d.requires_grad:
            m2d.retain_grad()

    def step_post_backward(
        self,
        params: Dict[str, Tensor],
        optimizers: Any,
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        lr: Optional[float] = None,
    ) -> None:
        """Accumulate grad stats, noise-inject, densify, and flush cache."""
        cfg = self.cfg
        densifier: HierarchicalMcmcDensifier = state["densifier"]
        cache: Optional[GpuSptCache] = state.get("cache")
        active_indices: Tensor = state["active_indices"]

        # --- 1. Screen-space gradient stats --------------------------------
        self._accumulate_grad_stats(info, active_indices, densifier, step)

        # --- 2. MCMC noise injection --------------------------------------
        # inject_noise_to_position uses a CUDA-only kernel
        # (quat_scale_to_covar_preci). Only fire on CUDA params; on CPU we
        # skip silently so CPU-only integration tests still run end-to-end.
        if lr is not None and cfg.T > 0 and step < cfg.lod_iters:
            means = params.get("means")
            if isinstance(means, torch.Tensor) and means.is_cuda:
                inject_noise_to_position(
                    params=params,
                    optimizers={},
                    state={},
                    scaler=lr * cfg.lr_noise_factor,
                )

        # --- 3. Densify ----------------------------------------------------
        do_densify = (
            step >= cfg.densify_start_iter
            and step < cfg.densify_stop_iter
            and (step - cfg.densify_start_iter) % cfg.densify_every == 0
        )
        if do_densify:
            n_spawn = self._compute_spawn_count(step, state)
            n_respawn = n_spawn // 2  # paper: roughly half as many respawns
            new_hspt = densifier.densify_step(step, n_spawn, n_respawn)
            state["hspt"] = new_hspt
            if cache is not None:
                # Topology changed → every cached cut is stale.
                cache.flush_all(writeback_cb=state.get("writeback_cb"))

        # --- 4. Periodic full cache flush ----------------------------------
        if (
            cache is not None
            and cfg.cache_full_flush_every > 0
            and step > 0
            and step % cfg.cache_full_flush_every == 0
            and not do_densify
        ):
            cache.flush_all(writeback_cb=state.get("writeback_cb"))

    # --- Internals ------------------------------------------------------

    @torch.no_grad()
    def _accumulate_grad_stats(
        self,
        info: Dict[str, Any],
        active_indices: Tensor,
        densifier: HierarchicalMcmcDensifier,
        step: int,
    ) -> None:
        """Fold per-Gaussian screen-space gradient magnitudes into the stats.

        The rasterizer's info dict commonly carries ``means2d`` of shape
        ``[C, N_render, 2]`` with a gradient we can read after backward. The
        screen-space gradient per Gaussian is taken as its L2 norm.

        If ``info`` has no suitable entry, this is a no-op.
        """
        m2d = info.get("means2d")
        if not isinstance(m2d, torch.Tensor) or m2d.grad is None:
            return
        grad = m2d.grad
        # grad: [C, N_render, 2] or [N_render, 2] — reduce camera dim and
        # take per-Gaussian L2 norm across xy.
        if grad.dim() == 3:
            grad = grad.abs().sum(dim=0)  # [N_render, 2]
        mag = grad.norm(dim=-1)  # [N_render]
        densifier.grad_stats.observe(active_indices, mag, step)

    def _compute_spawn_count(self, step: int, state: Dict[str, Any]) -> int:
        """Determine how many new Gaussians to spawn this densification pass.

        Paper suggests doubling every interval with a cap. We follow a simple
        linear schedule between start and stop, capped at a fraction of the
        remaining room in ``cfg.lod_cap_max``.
        """
        cfg = self.cfg
        remaining = max(0, cfg.lod_cap_max - state["store"].N)
        target_per_step = max(1, remaining // max(1, cfg.lod_iters // cfg.densify_every))
        return min(target_per_step, remaining)
