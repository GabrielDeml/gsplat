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

"""Out-of-core Adam.

Regular ``torch.optim.Adam`` tracks state tied to persistent parameter
tensors. In the LoD pipeline the GPU-resident parameters are ephemeral (a
different active subset each iteration), and the Adam state for all
Gaussians lives in host memory. This optimiser therefore:
  - pulls ``(exp_avg, exp_avg_sq, step)`` for the active subset from the
    ``CpuGaussianStore``
  - runs a standard Adam update on the GPU
  - scatters the updated state back to the CPU store.

Each Gaussian has its own step counter (incremented only when it's active)
so bias correction catches up smoothly regardless of streaming gaps.
"""

from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor

from .streaming import CpuGaussianStore


class OutOfCoreAdam:
    """Adam optimiser over a CpuGaussianStore with active-subset updates.

    Args:
        store: the CPU-resident Gaussian store (params + Adam state).
        lr_spec: per-parameter learning rates, e.g. ``{"means": 1e-4, ...}``.
        betas: Adam betas.
        eps: Adam epsilon.
        fused_param_keys: tuple of parameter keys to update. Keys missing
            from ``lr_spec`` are skipped.
    """

    def __init__(
        self,
        store: CpuGaussianStore,
        lr_spec: Dict[str, float],
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        fused_param_keys: Tuple[str, ...] = (
            "means",
            "scales",
            "quats",
            "opacities",
            "sh0",
            "shN",
        ),
    ):
        self.store = store
        self.lr_spec = dict(lr_spec)
        self.betas = betas
        self.eps = eps
        self.fused_param_keys = fused_param_keys

    @torch.no_grad()
    def step(
        self,
        active_indices: Tensor,
        active_params: Dict[str, Tensor],
    ) -> None:
        """Apply one Adam update.

        Args:
            active_indices: [M] long tensor of CpuGaussianStore indices.
            active_params: dict of GPU tensors whose ``.grad`` fields hold the
                current-iteration gradients. Each tensor's row ``i``
                corresponds to ``active_indices[i]``.
        """
        if active_indices.numel() == 0:
            return

        # Prepare adam state on the GPU for this subset.
        idx_cpu = active_indices.to("cpu")
        beta1, beta2 = self.betas
        updated_adam: Dict[str, Tensor] = {}
        updated_params: Dict[str, Tensor] = {}
        t_new: Optional[Tensor] = None

        for name in self.fused_param_keys:
            if name not in active_params:
                continue
            if name not in self.lr_spec:
                continue
            p = active_params[name]
            g = p.grad
            if g is None:
                continue

            device = p.device
            m = self.store.exp_avg[name][idx_cpu].to(device, non_blocking=True)
            v = self.store.exp_avg_sq[name][idx_cpu].to(device, non_blocking=True)

            # Step counter is shared across param names — compute once.
            if t_new is None:
                t = self.store.step[idx_cpu].to(device, non_blocking=True)
                t_new = (t + 1).to(torch.int64)

            m_new = beta1 * m + (1.0 - beta1) * g
            v_new = beta2 * v + (1.0 - beta2) * g.pow(2)

            # Bias correction with per-Gaussian step count.
            t_f = t_new.to(torch.float32)
            bc1 = 1.0 - beta1 ** t_f
            bc2 = 1.0 - beta2 ** t_f
            shape = [-1] + [1] * (m_new.dim() - 1)
            bc1_b = bc1.view(shape)
            bc2_b = bc2.view(shape)

            m_hat = m_new / bc1_b.clamp_min(1e-30)
            v_hat = v_new / bc2_b.clamp_min(1e-30)
            lr = float(self.lr_spec[name])
            update = lr * m_hat / (v_hat.sqrt() + self.eps)
            p_new = p.data - update

            updated_adam[f"exp_avg.{name}"] = m_new
            updated_adam[f"exp_avg_sq.{name}"] = v_new
            updated_params[name] = p_new

        if t_new is None:
            return  # no parameters updated

        updated_adam["step"] = t_new

        # Scatter back to host.
        self.store.scatter(idx_cpu, updated_params, adam_state=updated_adam)
