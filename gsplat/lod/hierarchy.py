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

"""Binary Gaussian hierarchy — the CPU-resident tree that backs the LoD system.

Layout:
  - Leaves and internal nodes share a single SoA tensor set indexed 0..N_total-1.
  - parent[i] = index of parent (-1 for root).
  - left[i], right[i] = children (-1 for leaf).
  - is_leaf[i] = True iff left[i] == right[i] == -1.
  - All per-node properties mirror gsplat's own storage conventions:
      scale stored as log-scale (activation = exp),
      opacity stored as logit  (activation = sigmoid),
      quat stored raw (activation = normalise).

The LoD metric m_d(i) uses activated scales:
    simple  : m_d(i) = T * max_j exp(scale_i^j)             (Eq. 3)
    refined : m_d(i) = T * sqrt(sum_{j!=k} s_i^j^2 s_i^k^2) (Eq. 6)
    where s_i^j = exp(scale_i^j).
Note: we use T * (size-like) so that m_d(child) < m_d(parent) holds under
standard moment-matched merging (parent has larger scale than children), which
is what the paper's BFS cut and SPT binary search require.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import Tensor


def _cpu(t: Tensor) -> Tensor:
    return t if t.device.type == "cpu" else t.detach().cpu()


@dataclass
class GaussianHierarchy:
    """A binary tree of Gaussians, CPU-resident, SoA storage.

    All tensors share dimension 0 = N_total = N_leaves + N_internal.
    Internal nodes come AFTER leaves in index space: indices [0, N_leaves) are
    leaves from the construction input order; indices [N_leaves, N_total) are
    internal nodes created during `build_hierarchy`.

    After densification, leaves may be converted to internal nodes; the
    invariant "leaves occupy the low indices" is NOT maintained long-term.
    Use ``is_leaf`` to check.
    """

    # Topology
    parent: Tensor  # [N_total] long, -1 for root
    left: Tensor  # [N_total] long, -1 for leaf
    right: Tensor  # [N_total] long, -1 for leaf
    is_leaf: Tensor  # [N_total] bool

    # Per-node Gaussian parameters (pre-activation)
    mu: Tensor  # [N_total, 3]
    scale: Tensor  # [N_total, 3]  log-scale
    quat: Tensor  # [N_total, 4]  raw (wxyz, will be normalised at render)
    opacity: Tensor  # [N_total]     logit
    sh0: Tensor  # [N_total, 1, 3]
    shN: Tensor  # [N_total, K, 3]  K = (sh_degree+1)^2 - 1; may be 0

    root: int  # index of the root node

    # Optional Gaussian-level metadata (populated lazily)
    subtree_size: Optional[Tensor] = None  # [N_total] int, number of leaves below
    depth: Optional[Tensor] = None  # [N_total] short, root=0

    # --- Basic invariants --------------------------------------------------

    @property
    def n_total(self) -> int:
        return int(self.parent.numel())

    @property
    def n_leaves(self) -> int:
        return int(self.is_leaf.sum().item())

    @property
    def n_internal(self) -> int:
        return self.n_total - self.n_leaves

    @property
    def sh_K(self) -> int:
        return int(self.shN.shape[1])

    # --- Derived quantities -----------------------------------------------

    def activated_scale(self, idx: Optional[Tensor] = None) -> Tensor:
        s = self.scale if idx is None else self.scale[idx]
        return torch.exp(s)

    def activated_opacity(self, idx: Optional[Tensor] = None) -> Tensor:
        o = self.opacity if idx is None else self.opacity[idx]
        return torch.sigmoid(o)

    def volume(self, idx: Optional[Tensor] = None) -> Tensor:
        """Volume of the Gaussian ellipsoid (s1 * s2 * s3), activated scales."""
        return self.activated_scale(idx).prod(dim=-1)

    def max_scale(self, idx: Optional[Tensor] = None) -> Tensor:
        return self.activated_scale(idx).amax(dim=-1)

    def compute_m_d(
        self, T: float, idx: Optional[Tensor] = None, refined: bool = False
    ) -> Tensor:
        """m_d per Eq. 3 (simple) or Eq. 6 (refined), computed on activated scales.

        Under moment-matched merging child m_d < parent m_d, which is the heap
        property required by the BFS cut and binary-search SPT cut.
        """
        s = self.activated_scale(idx)  # [..., 3]
        if not refined:
            return T * s.amax(dim=-1)
        s1, s2, s3 = s.unbind(-1)
        surface = torch.sqrt(
            s1 * s1 * s2 * s2 + s1 * s1 * s3 * s3 + s2 * s2 * s3 * s3 + 1e-30
        )
        return T * surface

    # --- Topology helpers --------------------------------------------------

    def sibling(self, idx: int) -> int:
        p = int(self.parent[idx].item())
        if p < 0:
            return -1
        l = int(self.left[p].item())
        r = int(self.right[p].item())
        return r if l == idx else l

    def compute_subtree_size(self) -> Tensor:
        """Fill and return ``subtree_size[i]`` = number of leaves below i.

        Processes nodes in order of decreasing depth so children are done first.
        """
        N = self.n_total
        size = torch.zeros(N, dtype=torch.int64)
        # Iterative post-order using in-degree counting over parents.
        # We compute sizes by iterating from leaves upward: each node's size is
        # sum of children's sizes (1 for a leaf).
        size[self.is_leaf] = 1
        # Build list of internal nodes sorted by depth (descending).
        if self.depth is None:
            self.depth = self._compute_depth()
        internal = torch.nonzero(~self.is_leaf, as_tuple=False).squeeze(-1)
        order = torch.argsort(self.depth[internal], descending=True)
        internal_sorted = internal[order]
        for i in internal_sorted.tolist():
            size[i] = size[self.left[i]] + size[self.right[i]]
        self.subtree_size = size
        return size

    def _compute_depth(self) -> Tensor:
        """BFS from root to populate depth (root = 0)."""
        N = self.n_total
        depth = torch.full((N,), -1, dtype=torch.int16)
        depth[self.root] = 0
        q = [self.root]
        while q:
            nxt = []
            for u in q:
                d = depth[u].item()
                for c in (int(self.left[u].item()), int(self.right[u].item())):
                    if c >= 0:
                        depth[c] = d + 1
                        nxt.append(c)
            q = nxt
        return depth

    # --- Export to gsplat-style params ------------------------------------

    def leaf_splats(self) -> Dict[str, Tensor]:
        """Return the leaves as a dict shaped like gsplat's splats ParameterDict.

        Keys: means, scales, quats, opacities, sh0, shN. Values live on CPU.
        """
        leaf_idx = torch.nonzero(self.is_leaf, as_tuple=False).squeeze(-1)
        return {
            "means": self.mu[leaf_idx].clone(),
            "scales": self.scale[leaf_idx].clone(),
            "quats": self.quat[leaf_idx].clone(),
            "opacities": self.opacity[leaf_idx].clone(),
            "sh0": self.sh0[leaf_idx].clone(),
            "shN": self.shN[leaf_idx].clone(),
        }

    def node_splats(self, indices: Tensor) -> Dict[str, Tensor]:
        """Return a subset of nodes in gsplat's splats shape."""
        return {
            "means": self.mu[indices],
            "scales": self.scale[indices],
            "quats": self.quat[indices],
            "opacities": self.opacity[indices],
            "sh0": self.sh0[indices],
            "shN": self.shN[indices],
        }

    # --- Mutation (used by densification) ---------------------------------

    def grow(self, n_new: int) -> Tensor:
        """Extend all SoA tensors by ``n_new`` entries. Returns the new indices.

        Newly allocated nodes are uninitialised: caller is responsible for
        assigning parent/left/right/is_leaf and all per-node properties.
        """
        old = self.n_total
        new = old + n_new

        def _grow(t: Tensor, fill=0) -> Tensor:
            shape = (new,) + tuple(t.shape[1:])
            out = torch.full(shape, fill, dtype=t.dtype) if fill != 0 else t.new_zeros(shape)
            out[:old] = t
            return out

        self.parent = _grow(self.parent, fill=-1)
        self.left = _grow(self.left, fill=-1)
        self.right = _grow(self.right, fill=-1)
        self.is_leaf = _grow(self.is_leaf, fill=False)
        self.mu = _grow(self.mu)
        self.scale = _grow(self.scale)
        self.quat = _grow(self.quat)
        self.opacity = _grow(self.opacity)
        self.sh0 = _grow(self.sh0)
        self.shN = _grow(self.shN)
        # Depth/subtree_size invalidated.
        self.subtree_size = None
        self.depth = None
        return torch.arange(old, new, dtype=torch.long)

    def convert_leaf_to_internal(self, node_idx: int, left_idx: int, right_idx: int) -> None:
        """Convert a leaf ``node_idx`` into an internal node with the given children."""
        assert self.is_leaf[node_idx], "convert_leaf_to_internal: target is not a leaf"
        self.is_leaf[node_idx] = False
        self.left[node_idx] = left_idx
        self.right[node_idx] = right_idx
        self.parent[left_idx] = node_idx
        self.parent[right_idx] = node_idx
        self.is_leaf[left_idx] = True
        self.is_leaf[right_idx] = True
        self.subtree_size = None
        self.depth = None

    def replace_child(self, parent_idx: int, old_child: int, new_child: int) -> None:
        """Replace one of ``parent_idx``'s children. Used when a dead leaf's
        parent is swallowed by its sibling."""
        if int(self.left[parent_idx].item()) == old_child:
            self.left[parent_idx] = new_child
        else:
            assert int(self.right[parent_idx].item()) == old_child
            self.right[parent_idx] = new_child
        self.parent[new_child] = parent_idx
        self.subtree_size = None
        self.depth = None

    # --- Validation --------------------------------------------------------

    def validate(self) -> None:
        """Raise if topology is inconsistent. Cheap; O(N)."""
        N = self.n_total
        parent = self.parent
        left = self.left
        right = self.right
        is_leaf = self.is_leaf
        # Exactly one root
        roots = torch.nonzero(parent < 0, as_tuple=False).squeeze(-1)
        assert roots.numel() == 1, f"expected 1 root, got {roots.numel()}"
        assert int(roots.item()) == self.root
        # Leaves have no children, internals have exactly 2 distinct children
        assert torch.all((left < 0) == is_leaf), "left-child vs is_leaf mismatch"
        assert torch.all((right < 0) == is_leaf), "right-child vs is_leaf mismatch"
        internal = ~is_leaf
        assert torch.all(left[internal] != right[internal]), "self-loop children"
        # Parent back-pointers are consistent
        left_i = left[internal]
        right_i = right[internal]
        internal_idx = torch.nonzero(internal, as_tuple=False).squeeze(-1)
        assert torch.all(parent[left_i] == internal_idx)
        assert torch.all(parent[right_i] == internal_idx)
