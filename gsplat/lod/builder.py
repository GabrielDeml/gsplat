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

"""Bottom-up Gaussian hierarchy builder.

Strategy: iterative mutual-nearest-neighbour pair merging.
 1. Start with the N input Gaussians as leaves (active).
 2. Build a KD-tree over active-node centres.
 3. For each active node, find its nearest other active node.
 4. Extract mutual pairs (i, nn[i]) with nn[nn[i]] == i and i < nn[i].
 5. Batched merge each mutual pair into a new internal node.
 6. Carry over unpaired nodes to the next round.
 7. Repeat until 1 node remains (the root).

The mutual-nearest pairing guarantees that at least one pair exists each
round (the global closest pair is always mutual) so termination is assured
in O(log N) rounds on typical inputs.

This is a pure-Python/PyTorch implementation intended for correctness and
debuggability on scenes up to a few million leaves. For >10M leaves a
CUDA port will eventually replace it (planned but deferred; see
``gsplat/cuda/csrc/lod_cut.cu`` in the plan).
"""

from typing import Dict

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor

from .hierarchy import GaussianHierarchy
from .merge import merge_pair_batch


def _to_cpu(d: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {k: v.detach().to("cpu") for k, v in d.items()}


@torch.no_grad()
def build_hierarchy(splats: Dict[str, Tensor]) -> GaussianHierarchy:
    """Build a binary Gaussian hierarchy from a splat dict.

    Args:
        splats: dict with keys ``means, scales, quats, opacities, sh0, shN``.
            Shapes: means [N, 3], scales [N, 3], quats [N, 4], opacities [N],
            sh0 [N, 1, 3], shN [N, K, 3] (K may be 0). Values can live on
            any device; they are moved to CPU.

    Returns:
        A fully populated ``GaussianHierarchy`` on CPU. If N == 1 the single
        input Gaussian is both leaf and root.
    """
    splats = _to_cpu(splats)
    N = splats["means"].shape[0]
    assert N > 0, "build_hierarchy: need at least one Gaussian"

    K_sh = splats["shN"].shape[1]

    # Pre-allocate slots for the final tree: worst case 2N - 1 nodes.
    N_total = max(1, 2 * N - 1)
    mu = splats["means"].new_zeros((N_total, 3))
    scale = splats["scales"].new_zeros((N_total, 3))
    quat = splats["quats"].new_zeros((N_total, 4))
    opacity = splats["opacities"].new_zeros((N_total,))
    sh0 = splats["sh0"].new_zeros((N_total, 1, 3))
    shN = splats["shN"].new_zeros((N_total, K_sh, 3))

    parent = torch.full((N_total,), -1, dtype=torch.long)
    left = torch.full((N_total,), -1, dtype=torch.long)
    right = torch.full((N_total,), -1, dtype=torch.long)
    is_leaf = torch.zeros(N_total, dtype=torch.bool)

    # Initialise leaves at indices [0, N).
    mu[:N] = splats["means"]
    scale[:N] = splats["scales"]
    quat[:N] = splats["quats"]
    opacity[:N] = splats["opacities"]
    sh0[:N] = splats["sh0"]
    shN[:N] = splats["shN"]
    is_leaf[:N] = True

    if N == 1:
        # Degenerate hierarchy: single node is both leaf and root.
        return GaussianHierarchy(
            parent=parent[:1],
            left=left[:1],
            right=right[:1],
            is_leaf=is_leaf[:1],
            mu=mu[:1],
            scale=scale[:1],
            quat=quat[:1],
            opacity=opacity[:1],
            sh0=sh0[:1],
            shN=shN[:1],
            root=0,
        )

    active = torch.arange(N, dtype=torch.long)
    next_slot = N

    while active.numel() > 1:
        positions = mu[active].numpy()

        if positions.shape[0] == 2:
            # Force-pair the last two.
            pair_a = torch.tensor([0], dtype=torch.long)
            pair_b = torch.tensor([1], dtype=torch.long)
        else:
            tree = cKDTree(positions)
            # Query k=2 (self + nearest other); pull index column 1.
            _, nn_idx = tree.query(positions, k=2)
            nn = torch.from_numpy(np.ascontiguousarray(nn_idx[:, 1])).long()  # [M]
            M = nn.numel()
            idx = torch.arange(M, dtype=torch.long)
            # Mutual pairs
            is_mutual = nn[nn] == idx
            keep = is_mutual & (idx < nn)
            pair_a = idx[keep]
            pair_b = nn[keep]

        if pair_a.numel() == 0:
            # Should not happen (the global closest pair is always mutual) but
            # guard against numerical ties: force-pair the first two actives.
            pair_a = torch.tensor([0], dtype=torch.long)
            pair_b = torch.tensor([1], dtype=torch.long)

        n_pairs = int(pair_a.numel())
        a_ids = active[pair_a]
        b_ids = active[pair_b]

        # Batched merge.
        merged = merge_pair_batch(
            {
                "means": mu[a_ids],
                "scales": scale[a_ids],
                "quats": quat[a_ids],
                "opacities": opacity[a_ids],
                "sh0": sh0[a_ids],
                "shN": shN[a_ids],
            },
            {
                "means": mu[b_ids],
                "scales": scale[b_ids],
                "quats": quat[b_ids],
                "opacities": opacity[b_ids],
                "sh0": sh0[b_ids],
                "shN": shN[b_ids],
            },
        )

        # Assign new internal-node slots.
        new_slots = torch.arange(next_slot, next_slot + n_pairs, dtype=torch.long)
        next_slot += n_pairs

        mu[new_slots] = merged["means"]
        scale[new_slots] = merged["scales"]
        quat[new_slots] = merged["quats"]
        opacity[new_slots] = merged["opacities"]
        sh0[new_slots] = merged["sh0"]
        shN[new_slots] = merged["shN"]

        parent[a_ids] = new_slots
        parent[b_ids] = new_slots
        left[new_slots] = a_ids
        right[new_slots] = b_ids
        is_leaf[new_slots] = False

        # Build next active set: new slots + unpaired.
        paired_mask = torch.zeros(active.numel(), dtype=torch.bool)
        paired_mask[pair_a] = True
        paired_mask[pair_b] = True
        unpaired = active[~paired_mask]
        active = torch.cat([new_slots, unpaired])

    # Trim allocations: we may have used fewer than 2N - 1 slots if duplicates
    # forced odd pairings, but with the force-pair fallback above we allocate
    # exactly one internal per pair. In the mutual-NN path we still use exactly
    # N - 1 internals total across rounds.
    used = next_slot
    if used < N_total:
        mu = mu[:used].contiguous()
        scale = scale[:used].contiguous()
        quat = quat[:used].contiguous()
        opacity = opacity[:used].contiguous()
        sh0 = sh0[:used].contiguous()
        shN = shN[:used].contiguous()
        parent = parent[:used].contiguous()
        left = left[:used].contiguous()
        right = right[:used].contiguous()
        is_leaf = is_leaf[:used].contiguous()

    root = int(active.item())

    h = GaussianHierarchy(
        parent=parent,
        left=left,
        right=right,
        is_leaf=is_leaf,
        mu=mu,
        scale=scale,
        quat=quat,
        opacity=opacity,
        sh0=sh0,
        shN=shN,
        root=root,
    )
    return h
