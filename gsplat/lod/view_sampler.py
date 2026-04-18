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

"""k-NN view sampler (paper §4.4).

Precompute a k-nearest-neighbour graph over the training view positions. For
successive training iterations, sample the next view from the neighbours of
the current one with probability ``P(j | i) = 1 / (w_ij + W)`` (normalised),
where ``w_ij`` is the Euclidean distance. Every ``uniform_every`` iterations,
draw a uniform view instead (to prevent overfitting to local spatial clusters).
"""

from typing import Optional

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor


class KnnViewSampler:
    """Stateful view sampler driven by a k-NN graph.

    Args:
        view_positions: [V, 3] tensor of camera centres.
        k: number of neighbours to keep per view (includes the view itself, so
            effective pool = k - 1; paper uses k=32).
        W: smoothing constant in ``1 / (w + W)`` (paper uses 1.0).
        uniform_every: periodicity of uniform fallback (paper uses 128).
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        view_positions: Tensor,
        k: int = 32,
        W: float = 1.0,
        uniform_every: int = 128,
        seed: int = 0,
    ):
        assert view_positions.dim() == 2 and view_positions.shape[1] == 3
        self.V = int(view_positions.shape[0])
        self.k = min(int(k), self.V)
        self.W = float(W)
        self.uniform_every = int(uniform_every)
        self._rng = np.random.default_rng(seed)

        pos_np = view_positions.detach().cpu().numpy().astype(np.float32)
        tree = cKDTree(pos_np)
        dists, idx = tree.query(pos_np, k=self.k)  # dists[v, 0] = 0 (self)
        # Keep neighbours excluding the self index when present.
        self._neighbours = idx[:, 1:] if self.k > 1 else idx
        self._distances = dists[:, 1:] if self.k > 1 else dists

        # Precompute sampling probabilities per row.
        w = self._distances.astype(np.float64) + self.W
        probs = 1.0 / w
        probs = probs / probs.sum(axis=1, keepdims=True).clip(min=1e-30)
        self._probs = probs  # [V, k-1]

    # --- Sampling --------------------------------------------------------

    def sample_next(self, current_idx: int, iter_step: int) -> int:
        """Pick the next training view index given the current and step."""
        if self.uniform_every > 0 and (iter_step % self.uniform_every) == 0:
            return int(self._rng.integers(0, self.V))
        if self._neighbours.shape[1] == 0:
            # Degenerate: single view.
            return int(current_idx)
        probs = self._probs[current_idx]
        choice = int(self._rng.choice(self._neighbours.shape[1], p=probs))
        return int(self._neighbours[current_idx, choice])

    # --- Accessors for tests ---------------------------------------------

    def neighbours(self, view_idx: int) -> np.ndarray:
        return np.asarray(self._neighbours[view_idx])

    def distribution(self, view_idx: int) -> np.ndarray:
        return np.asarray(self._probs[view_idx])
