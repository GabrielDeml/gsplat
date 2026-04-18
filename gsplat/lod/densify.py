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

"""Hierarchical MCMC densifier.

Paper §4.2 + Fig. 2b. Two operations on the ``GaussianHierarchy``:

    spawn(leaf c):
        Convert c from leaf to internal node. Allocate two new leaves
        c_L, c_R with perturbed positions / scaled covariances (the standard
        3DGS-MCMC split). Parent c's params remain valid as the merged
        approximation of its two new children.

    respawn(dead leaf d):
        Let p = parent(d), s = sibling(d).
        Replace p in the tree by s (grandparent now points to s; d and the
        node formerly holding p are freed).
        Pair (d, ex-p) is then re-attached as children of a high-grad leaf
        via ``spawn``-style initialisation, similar to the paper's
        "Gaussians respawned as children to another node".

Gradient statistics (max screen-space grad per Gaussian) are maintained as
a running update. Opacity-dead leaves and unused-long-enough leaves (paper
§A.5 pruning) are candidates for respawn.

This module operates entirely on CPU structures — the GPU is only used for
per-iteration gradient accumulation, which the strategy forwards here.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .hierarchy import GaussianHierarchy
from .hspt import HSPT, build_hspt
from .streaming import CpuGaussianStore


@dataclass
class GradStats:
    """Running max-absolute screen-space gradient per CpuGaussianStore index.

    Maintained on CPU. Updated from per-iteration info dicts.
    """

    max_grad: Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.float32))
    last_seen: Tensor = field(default_factory=lambda: torch.full((0,), -1, dtype=torch.int64))

    def resize(self, n: int) -> None:
        if n <= self.max_grad.numel():
            return
        new_max = torch.zeros(n, dtype=torch.float32)
        new_max[: self.max_grad.numel()] = self.max_grad
        new_seen = torch.full((n,), -1, dtype=torch.int64)
        new_seen[: self.last_seen.numel()] = self.last_seen
        self.max_grad = new_max
        self.last_seen = new_seen

    @torch.no_grad()
    def observe(self, indices: Tensor, grads: Tensor, step: int) -> None:
        """Fold in per-Gaussian grad magnitudes.

        Args:
            indices: [M] long (CpuGaussianStore indices; CPU or GPU ok).
            grads: [M] float (>= 0 — caller should pass L2-norm of means2d grad).
            step: iteration index.
        """
        if indices.numel() == 0:
            return
        self.resize(int(indices.max().item()) + 1)
        idx = indices.to("cpu")
        g = grads.detach().to("cpu").abs()
        current = self.max_grad[idx]
        self.max_grad[idx] = torch.maximum(current, g)
        self.last_seen[idx] = step

    def reset(self, indices: Tensor) -> None:
        idx = indices.to("cpu")
        self.max_grad[idx] = 0.0


# -----------------------------------------------------------------------------


@dataclass
class HierarchicalMcmcDensifier:
    """Coordinator for spawn/respawn densification over the hierarchy.

    Usage:

        densifier = HierarchicalMcmcDensifier(cfg, hierarchy, store, hspt, cache)
        # each iteration:
        densifier.grad_stats.observe(active_indices, means2d_grad_norm, step)
        # periodically:
        hspt_new = densifier.densify_step(step, n_spawn, n_respawn)

    Attributes:
        cfg, hierarchy, store, hspt, cache: wired from the trainer.
        grad_stats: per-Gaussian max-grad tracker.
    """

    hierarchy: GaussianHierarchy
    store: CpuGaussianStore
    hspt_size_threshold: float
    respawn_opacity_threshold: float = 0.005
    respawn_unused_iters: int = 600
    spawn_position_noise: float = 0.5  # fraction of scale
    spawn_scale_shrink: float = 1.6  # child scale = parent scale / shrink (MCMC split)
    refined_m_d: bool = False

    grad_stats: GradStats = field(default_factory=GradStats)

    # --- Primary API -----------------------------------------------------

    @torch.no_grad()
    def densify_step(
        self,
        step: int,
        n_spawn: int,
        n_respawn: int,
    ) -> HSPT:
        """Perform one densification pass. Returns a freshly-built HSPT.

        The caller is responsible for invalidating any GPU caches that
        depended on the old HSPT — topology changes make entries stale.
        """
        h = self.hierarchy
        self.grad_stats.resize(self.store.N)

        # --- 1. Identify respawn candidates (dead / unused leaves) ----------
        opacity = self.store.opacities[: self.store.N]
        is_leaf_live = self._live_leaf_mask()
        dead = (
            torch.sigmoid(opacity) < self.respawn_opacity_threshold
        ) & is_leaf_live

        if self.respawn_unused_iters > 0:
            seen = self.grad_stats.last_seen[: self.store.N]
            unused = ((step - seen) > self.respawn_unused_iters) & (seen >= 0) & is_leaf_live
            dead = dead | unused

        dead_ids = torch.nonzero(dead, as_tuple=False).squeeze(-1)
        if n_respawn < dead_ids.numel():
            # Pick the n_respawn worst (lowest opacity) to actually respawn.
            order = torch.argsort(torch.sigmoid(opacity[dead_ids]))
            dead_ids = dead_ids[order[:n_respawn]]

        # --- 2. Identify spawn candidates (top-k screen-space grad leaves) --
        grad = self.grad_stats.max_grad[: self.store.N]
        scores = torch.where(is_leaf_live, grad, torch.zeros_like(grad))
        # Exclude dead leaves (they're respawning, not spawning).
        if dead_ids.numel() > 0:
            scores[dead_ids] = 0.0
        n_spawn = min(n_spawn, int((scores > 0).sum().item()))
        if n_spawn > 0:
            spawn_ids = torch.topk(scores, n_spawn).indices
        else:
            spawn_ids = torch.empty(0, dtype=torch.long)

        # --- 3. Execute spawns ---------------------------------------------
        for c in spawn_ids.tolist():
            self._spawn(int(c))

        # --- 4. Execute respawns -------------------------------------------
        # Extra targets for respawn placement: another top-grad slice.
        extra_targets = self._pick_respawn_targets(dead_ids.numel(), exclude=spawn_ids)
        for d, target in zip(dead_ids.tolist(), extra_targets.tolist()):
            self._respawn(int(d), int(target))

        # Reset grad stats for touched indices so we don't respawn on stale grads.
        if spawn_ids.numel() > 0:
            self.grad_stats.reset(spawn_ids)
        if dead_ids.numel() > 0:
            self.grad_stats.reset(dead_ids)

        # --- 5. Rebuild HSPT -----------------------------------------------
        return build_hspt(h, size=self.hspt_size_threshold, refined_m_d=self.refined_m_d)

    # --- Operations ----------------------------------------------------

    def _spawn(self, c: int) -> None:
        """Convert leaf c into an internal node with two new, perturbed leaves."""
        h = self.hierarchy
        store = self.store
        assert bool(h.is_leaf[c]), f"spawn: node {c} is not a leaf"

        # Allocate two new rows in the CPU store and the hierarchy.
        new_ids = store.allocate_new(2)  # CPU store indices
        h_grow = h.grow(2)  # hierarchy node ids (==store indices by construction)
        # In this scheme, hierarchy node ids are CpuGaussianStore indices
        # 1:1 (both row 0..N_total-1). build_hierarchy already populates
        # entries 0..N-1 as leaves with the initial splats, and Nth+ with
        # internals. For densification we add new leaves *after* the current
        # hierarchy size in lockstep with the store.
        # The simplest invariant to maintain is: hierarchy index == store idx.
        # We check that invariant:
        assert new_ids.tolist() == h_grow.tolist(), "store/hierarchy index drift"

        l_id = int(new_ids[0].item())
        r_id = int(new_ids[1].item())

        # Seed both children from parent's params with a small position
        # perturbation along a random axis scaled by parent's max scale.
        parent_mu = store.means[c]
        parent_scale = store.scales[c]
        max_scale = float(torch.exp(parent_scale).amax().item())
        perturb = torch.randn(3) * (self.spawn_position_noise * max_scale)

        mu_L = parent_mu + perturb
        mu_R = parent_mu - perturb
        # Shrink the children's scales: log(s/shrink) = log(s) - log(shrink).
        shrink = torch.log(torch.tensor(float(self.spawn_scale_shrink)))
        scale_LR = parent_scale - shrink
        # Share parent's quat and SH, halve opacity so union stays near parent's.
        parent_sig = float(torch.sigmoid(store.opacities[c]).item())
        half_sig = 1.0 - (1.0 - parent_sig) ** 0.5  # so union of 2 equals parent
        half_logit = torch.log(torch.tensor(half_sig / (1.0 - half_sig + 1e-6) + 1e-6))

        for child_id, mu in ((l_id, mu_L), (r_id, mu_R)):
            store.means[child_id] = mu
            store.scales[child_id] = scale_LR
            store.quats[child_id] = store.quats[c].clone()
            store.opacities[child_id] = half_logit
            store.sh0[child_id] = store.sh0[c].clone()
            store.shN[child_id] = store.shN[c].clone()
            # Zero Adam state for new children.
            for k in store.exp_avg.keys():
                store.exp_avg[k][child_id].zero_()
                store.exp_avg_sq[k][child_id].zero_()
            store.step[child_id] = 0

        # Populate hierarchy arrays for the new entries (grow() zeroed them).
        h.mu[l_id] = store.means[l_id]
        h.mu[r_id] = store.means[r_id]
        h.scale[l_id] = store.scales[l_id]
        h.scale[r_id] = store.scales[r_id]
        h.quat[l_id] = store.quats[l_id]
        h.quat[r_id] = store.quats[r_id]
        h.opacity[l_id] = store.opacities[l_id]
        h.opacity[r_id] = store.opacities[r_id]
        h.sh0[l_id] = store.sh0[l_id]
        h.sh0[r_id] = store.sh0[r_id]
        h.shN[l_id] = store.shN[l_id]
        h.shN[r_id] = store.shN[r_id]

        # Convert c from leaf to internal.
        h.convert_leaf_to_internal(c, l_id, r_id)

    def _respawn(self, d: int, target: int) -> None:
        """Remove dead leaf d, swallow its parent into its sibling, then attach
        (d, ex_parent) as two new leaves under `target` (which must currently
        be a leaf).

        If d is the root (N_leaves == 1 edge case) we skip.
        """
        h = self.hierarchy
        store = self.store
        if int(h.parent[d].item()) < 0:
            return
        p = int(h.parent[d].item())
        s = h.sibling(d)
        if s < 0:
            return

        # grandparent -> sibling
        gp = int(h.parent[p].item())
        if gp < 0:
            # p was the root: sibling becomes the new root.
            h.parent[s] = -1
            h.root = s
        else:
            h.replace_child(gp, p, s)

        # d and p are now orphaned. Mark both as leaves and place them under
        # target via the spawn pathway — but for that they need to be at the
        # end of the hierarchy / store arrays. We keep the hierarchy == store
        # index invariant and simply reuse these existing rows.
        # To "attach under target", we first make target an internal node
        # with d and p as its children.
        assert bool(h.is_leaf[target]), "respawn: target must be a leaf"
        assert target != d and target != p

        # Clear any children the orphans might still point to.
        h.left[d] = -1
        h.right[d] = -1
        h.is_leaf[d] = True
        h.left[p] = -1
        h.right[p] = -1
        h.is_leaf[p] = True

        # Re-initialise d and p at the target's location with MCMC-style
        # perturbation.
        target_mu = store.means[target]
        target_scale = store.scales[target]
        max_scale = float(torch.exp(target_scale).amax().item())
        perturb = torch.randn(3) * (0.5 * max_scale)
        shrink = torch.log(torch.tensor(float(self.spawn_scale_shrink)))
        new_scale = target_scale - shrink
        target_sig = float(torch.sigmoid(store.opacities[target]).item())
        half_sig = 1.0 - (1.0 - target_sig) ** 0.5
        half_logit = torch.log(torch.tensor(half_sig / (1.0 - half_sig + 1e-6) + 1e-6))

        for child_id, mu in ((d, target_mu + perturb), (p, target_mu - perturb)):
            store.means[child_id] = mu
            store.scales[child_id] = new_scale
            store.quats[child_id] = store.quats[target].clone()
            store.opacities[child_id] = half_logit
            store.sh0[child_id] = store.sh0[target].clone()
            store.shN[child_id] = store.shN[target].clone()
            for k in store.exp_avg.keys():
                store.exp_avg[k][child_id].zero_()
                store.exp_avg_sq[k][child_id].zero_()
            store.step[child_id] = 0
            # Mirror into hierarchy arrays.
            h.mu[child_id] = store.means[child_id]
            h.scale[child_id] = store.scales[child_id]
            h.quat[child_id] = store.quats[child_id]
            h.opacity[child_id] = store.opacities[child_id]
            h.sh0[child_id] = store.sh0[child_id]
            h.shN[child_id] = store.shN[child_id]

        # Attach d, p as children of target.
        h.convert_leaf_to_internal(target, d, p)

    # --- Helpers --------------------------------------------------------

    def _live_leaf_mask(self) -> Tensor:
        """Bool mask over the store's live indices: True iff the corresponding
        hierarchy node is a leaf. Assumes the hierarchy is at least as large
        as the store.
        """
        N = self.store.N
        h = self.hierarchy
        if h.n_total >= N:
            return h.is_leaf[:N].clone()
        out = torch.zeros(N, dtype=torch.bool)
        out[: h.n_total] = h.is_leaf
        return out

    def _pick_respawn_targets(self, n: int, exclude: Optional[Tensor] = None) -> Tensor:
        """Pick ``n`` leaves with high screen-space grad to receive respawned
        children. Targets must be leaves. Duplicates are allowed iff n > candidate count.
        """
        if n == 0:
            return torch.empty(0, dtype=torch.long)
        N = self.store.N
        grad = self.grad_stats.max_grad[:N].clone()
        live_leaf = self._live_leaf_mask()
        grad[~live_leaf] = 0.0
        if exclude is not None and exclude.numel() > 0:
            grad[exclude] = 0.0
        # Take top-n; if fewer nonzero, wrap with replacement.
        nonzero = int((grad > 0).sum().item())
        if nonzero >= n:
            return torch.topk(grad, n).indices
        # Fewer candidates than needed — wrap with replacement over the
        # top-nonzero set.
        if nonzero == 0:
            # Fall back: random leaves.
            leaf_idx = torch.nonzero(live_leaf, as_tuple=False).squeeze(-1)
            if leaf_idx.numel() == 0:
                return torch.empty(0, dtype=torch.long)
            return leaf_idx[torch.randint(0, leaf_idx.numel(), (n,))]
        top = torch.topk(grad, nonzero).indices
        rep = (n + nonzero - 1) // nonzero
        out = top.repeat(rep)[:n]
        return out
