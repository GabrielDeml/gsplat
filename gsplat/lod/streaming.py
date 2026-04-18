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

"""CPU-resident Gaussian store for out-of-core training.

The ``CpuGaussianStore`` holds the full Gaussian parameter set plus its Adam
optimiser state on the host. Each training iteration pulls a subset to the
GPU via ``gather``, computes gradients, and scatters updates back via
``scatter``.

This module intentionally does NOT implement pinned memory (paper §A.4).
Adding it later is a two-line change: `pin_memory=True` on the underlying
tensors.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
from torch import Tensor


_PARAM_KEYS = ("means", "scales", "quats", "opacities", "sh0", "shN")


@dataclass
class CpuGaussianStore:
    """Struct-of-arrays storage of all Gaussians + their Adam state on CPU.

    Attributes (all tensors on CPU, float32 params / moments):

        means     [N, 3]
        scales    [N, 3]        log-scale
        quats     [N, 4]        raw wxyz (normalised at rendering)
        opacities [N]           logit
        sh0       [N, 1, 3]
        shN       [N, K, 3]     K = (sh_degree+1)^2 - 1

        exp_avg[name]    same-shape as the corresponding param
        exp_avg_sq[name] same-shape as the corresponding param
        step             [N] int64 — per-Gaussian Adam step counter

    ``capacity`` is the allocated size (>= current N). ``N`` is the active
    population. Indices in [0, N) are "live". Beyond N is preallocated slack
    that densification can fill via :meth:`allocate_new`.

    Morton-order reordering (paper §A.5) is not implemented yet: it can be
    added as a one-shot ``reorder`` method once densification lands.
    """

    means: Tensor
    scales: Tensor
    quats: Tensor
    opacities: Tensor
    sh0: Tensor
    shN: Tensor
    exp_avg: Dict[str, Tensor]
    exp_avg_sq: Dict[str, Tensor]
    step: Tensor
    N: int
    capacity: int

    # --- Construction ------------------------------------------------------

    @classmethod
    def from_splats(
        cls,
        splats: Dict[str, Tensor],
        capacity: Optional[int] = None,
    ) -> "CpuGaussianStore":
        """Build a store from a dict of initial splats.

        Args:
            splats: keys ``means, scales, quats, opacities, sh0, shN``, any
                device. Values are detached and moved to CPU.
            capacity: if given, allocate for this many Gaussians (>= N). If
                omitted, capacity = N (no slack).
        """
        N = int(splats["means"].shape[0])
        capacity = int(capacity) if capacity is not None else N
        assert capacity >= N, f"capacity {capacity} < N {N}"
        K_sh = int(splats["shN"].shape[1])

        def _alloc(shape, dtype=torch.float32, fill=0.0):
            t = torch.empty(shape, dtype=dtype)
            t.zero_()
            if fill != 0.0:
                t.fill_(fill)
            return t

        means = _alloc((capacity, 3))
        scales = _alloc((capacity, 3))
        quats = _alloc((capacity, 4))
        opacities = _alloc((capacity,))
        sh0 = _alloc((capacity, 1, 3))
        shN = _alloc((capacity, K_sh, 3))

        # Fill the live slice.
        for src, dst in zip(
            (splats["means"], splats["scales"], splats["quats"], splats["opacities"],
             splats["sh0"], splats["shN"]),
            (means, scales, quats, opacities, sh0, shN),
        ):
            dst[:N] = src.detach().to("cpu")

        exp_avg = {
            "means": torch.zeros_like(means),
            "scales": torch.zeros_like(scales),
            "quats": torch.zeros_like(quats),
            "opacities": torch.zeros_like(opacities),
            "sh0": torch.zeros_like(sh0),
            "shN": torch.zeros_like(shN),
        }
        exp_avg_sq = {k: torch.zeros_like(v) for k, v in exp_avg.items()}
        step = torch.zeros((capacity,), dtype=torch.int64)

        return cls(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            step=step,
            N=N,
            capacity=capacity,
        )

    # --- Access helpers ----------------------------------------------------

    def _param_tensor(self, name: str) -> Tensor:
        return getattr(self, name)

    def gather(
        self,
        indices: Tensor,
        device: str | torch.device = "cuda",
        copy_adam: bool = False,
    ) -> Dict[str, Tensor]:
        """Pull parameters for ``indices`` to ``device``.

        Returns a dict with the 6 param keys. If ``copy_adam``, also returns
        keys ``exp_avg.{name}``, ``exp_avg_sq.{name}``, and ``step``.
        """
        out: Dict[str, Tensor] = {}
        idx = indices.to("cpu")
        for k in _PARAM_KEYS:
            out[k] = self._param_tensor(k)[idx].to(device, non_blocking=True)
        if copy_adam:
            for k in _PARAM_KEYS:
                out[f"exp_avg.{k}"] = self.exp_avg[k][idx].to(device, non_blocking=True)
                out[f"exp_avg_sq.{k}"] = self.exp_avg_sq[k][idx].to(device, non_blocking=True)
            out["step"] = self.step[idx].to(device, non_blocking=True)
        return out

    @torch.no_grad()
    def scatter(
        self,
        indices: Tensor,
        params: Dict[str, Tensor],
        adam_state: Optional[Dict[str, Tensor]] = None,
    ) -> None:
        """Write updated params back to host. ``indices`` selects which rows.

        ``params`` keys match those returned by :meth:`gather` (without Adam
        prefixes). If ``adam_state`` is given with keys matching the
        ``exp_avg.<name>`` / ``exp_avg_sq.<name>`` / ``step`` convention from
        :meth:`gather`, Adam moments and step counter are updated too.
        """
        idx = indices.to("cpu")
        for k in _PARAM_KEYS:
            if k in params:
                self._param_tensor(k)[idx] = params[k].detach().to("cpu")
        if adam_state is not None:
            for k in _PARAM_KEYS:
                ek = f"exp_avg.{k}"
                esk = f"exp_avg_sq.{k}"
                if ek in adam_state:
                    self.exp_avg[k][idx] = adam_state[ek].detach().to("cpu")
                if esk in adam_state:
                    self.exp_avg_sq[k][idx] = adam_state[esk].detach().to("cpu")
            if "step" in adam_state:
                self.step[idx] = adam_state["step"].detach().to("cpu")

    # --- Growth / pruning --------------------------------------------------

    @torch.no_grad()
    def allocate_new(self, n: int) -> Tensor:
        """Reserve ``n`` new indices. If capacity is exceeded, the underlying
        tensors are grown (at the cost of a full reallocation)."""
        needed = self.N + n
        if needed > self.capacity:
            new_capacity = max(needed, int(self.capacity * 1.5) + 1)
            self._grow_to(new_capacity)
        new_idx = torch.arange(self.N, self.N + n, dtype=torch.long)
        self.N += n
        return new_idx

    def _grow_to(self, new_capacity: int) -> None:
        def _realloc(t: Tensor) -> Tensor:
            shape = (new_capacity,) + tuple(t.shape[1:])
            out = t.new_zeros(shape)
            out[: t.shape[0]] = t
            return out

        self.means = _realloc(self.means)
        self.scales = _realloc(self.scales)
        self.quats = _realloc(self.quats)
        self.opacities = _realloc(self.opacities)
        self.sh0 = _realloc(self.sh0)
        self.shN = _realloc(self.shN)
        for k in list(self.exp_avg.keys()):
            self.exp_avg[k] = _realloc(self.exp_avg[k])
            self.exp_avg_sq[k] = _realloc(self.exp_avg_sq[k])
        self.step = _realloc(self.step)
        self.capacity = new_capacity

    # --- Stats / sizes -----------------------------------------------------

    def bytes_per_gaussian(self) -> int:
        """Estimated CPU bytes per live Gaussian (params + Adam)."""
        K_sh = int(self.shN.shape[1])
        params_floats = 3 + 3 + 4 + 1 + 3 + K_sh * 3
        adam_floats = 2 * params_floats  # exp_avg + exp_avg_sq
        step_bytes = 8  # int64
        return (params_floats + adam_floats) * 4 + step_bytes

    def total_bytes(self) -> int:
        return self.bytes_per_gaussian() * self.N
