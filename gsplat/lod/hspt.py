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

"""Hierarchical Sequential Point Trees (HSPT).

Paper §4.3: cut the ``GaussianHierarchy`` at a volume threshold ``size``. Nodes
with ``volume > size`` form the *upper hierarchy* (a tree traversed with a BFS
cut). Each subtree rooted at a child of the cut set whose volume <= size becomes
an independent *SPT*, stored as a flat array of entries sorted by
``m_d(parent)`` in descending order. SPT cuts are then one binary search on
``m_d(parent)`` vs the camera-to-SPT-root distance.

Rebuilt only after a densification step; the per-iteration cut code does not
mutate the HSPT.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import Tensor

from .hierarchy import GaussianHierarchy


@dataclass
class HSPT:
    """A hierarchical SPT layered over a ``GaussianHierarchy``.

    Upper arrays index nodes in the *hierarchy* (not a separate renumbering).
    For each upper node we record its children within the upper set, or an
    SPT id if the child has volume <= size and thus lives in an SPT.

    SPT entries are concatenated CSR-style. For SPT ``s``:
        entries = spt_entries_node_id[spt_offset[s]:spt_offset[s+1]]
    sorted by ``spt_entries_md_parent[...]`` DESCENDING. The sort order enables
    binary search at cut time: given a camera distance d, the cut is the
    prefix of entries with ``md_parent > d``.
    """

    # Reference hierarchy (CPU, not owned).
    hierarchy: GaussianHierarchy

    # --- Upper hierarchy --------------------------------------------------
    upper_ids: Tensor  # [U] long (node ids in the hierarchy)
    upper_is_root: Tensor  # [U] bool, True for the global root entry
    upper_left: Tensor  # [U] long; -1 if child is an SPT root or out-of-upper
    upper_right: Tensor  # [U] long
    upper_left_spt: Tensor  # [U] long; spt_id if upper_left is -1 but subtree exists
    upper_right_spt: Tensor  # [U] long; spt_id if upper_right is -1 but subtree exists

    # Inverse map: hierarchy node id -> upper idx (-1 if not in upper).
    node_to_upper: Tensor  # [N_total] long, -1 outside upper

    # --- SPTs --------------------------------------------------------------
    spt_root_node_id: Tensor  # [S] long (hierarchy index of each SPT's root)
    spt_offset: Tensor  # [S + 1] long (CSR offsets into spt_entries_*)
    spt_entries_node_id: Tensor  # [E] long (hierarchy index per entry)
    # Stored WITHOUT the T factor: the size-like quantity q such that
    # m_d = T * q. This way the same HSPT is valid for any T.
    # Sort order within each SPT is DESCENDING in q.
    spt_entries_size_parent: Tensor  # [E] float, sort key (descending within each SPT)
    # Per-entry SPT membership, used by the vectorised GPU cut path.
    spt_entries_spt_id: Tensor  # [E] long

    # Size threshold used at build time (for debugging).
    size_threshold: float = 0.0

    # Precomputed upper-level aggregates used by the cut code.
    upper_mu: Optional[Tensor] = None  # [U, 3]
    upper_max_scale: Optional[Tensor] = None  # [U]
    upper_m_d: Optional[Tensor] = None  # [U]

    # --- API --------------------------------------------------------------

    @property
    def n_upper(self) -> int:
        return int(self.upper_ids.numel())

    @property
    def n_spts(self) -> int:
        return int(self.spt_root_node_id.numel())

    @property
    def n_entries(self) -> int:
        return int(self.spt_entries_node_id.numel())

    def spt_range(self, spt_id: int):
        return int(self.spt_offset[spt_id].item()), int(self.spt_offset[spt_id + 1].item())

    def precompute_upper_metrics(self, T: float, refined: bool = False) -> None:
        """Fill upper_mu / upper_max_scale / upper_m_d for the current T."""
        h = self.hierarchy
        self.upper_mu = h.mu[self.upper_ids].clone()
        self.upper_max_scale = h.max_scale(self.upper_ids).clone()
        self.upper_m_d = h.compute_m_d(T, self.upper_ids, refined=refined).clone()

    def to(self, device) -> "HSPT":
        """Return a copy with all topology/metadata tensors on ``device``.

        The referenced ``GaussianHierarchy`` stays on CPU (it's the source of
        truth for out-of-core storage); only HSPT's own arrays migrate, since
        those are what the GPU cut path consumes.
        """
        return HSPT(
            hierarchy=self.hierarchy,
            upper_ids=self.upper_ids.to(device),
            upper_is_root=self.upper_is_root.to(device),
            upper_left=self.upper_left.to(device),
            upper_right=self.upper_right.to(device),
            upper_left_spt=self.upper_left_spt.to(device),
            upper_right_spt=self.upper_right_spt.to(device),
            node_to_upper=self.node_to_upper.to(device),
            spt_root_node_id=self.spt_root_node_id.to(device),
            spt_offset=self.spt_offset.to(device),
            spt_entries_node_id=self.spt_entries_node_id.to(device),
            spt_entries_size_parent=self.spt_entries_size_parent.to(device),
            spt_entries_spt_id=self.spt_entries_spt_id.to(device),
            size_threshold=self.size_threshold,
            upper_mu=None if self.upper_mu is None else self.upper_mu.to(device),
            upper_max_scale=(
                None if self.upper_max_scale is None else self.upper_max_scale.to(device)
            ),
            upper_m_d=None if self.upper_m_d is None else self.upper_m_d.to(device),
        )


def _collect_subtree(h: GaussianHierarchy, root: int) -> List[int]:
    """Iterative DFS returning all node ids below (and including) ``root``."""
    out: List[int] = []
    stack: List[int] = [root]
    while stack:
        n = stack.pop()
        out.append(n)
        l = int(h.left[n].item())
        r = int(h.right[n].item())
        if l >= 0:
            stack.append(l)
        if r >= 0:
            stack.append(r)
    return out


@torch.no_grad()
def build_hspt(h: GaussianHierarchy, size: float, refined_m_d: bool = False) -> HSPT:
    """Build an HSPT from a ``GaussianHierarchy``.

    Args:
        h: the hierarchy.
        size: volume threshold separating upper from lower. Nodes with
            ``volume > size`` are in the upper set; their children whose
            ``volume <= size`` become SPT roots.
        refined_m_d: if True use Eq. 6 (sqrt of surface-area-like quantity)
            for the stored "size" sort key; else Eq. 3 (max scale component).
            The choice must match the cut-time m_d formula.

    Returns:
        A populated ``HSPT``. Entries are sorted descending by the T-free
        size quantity; the cut code multiplies by T on the fly.
    """
    N = h.n_total
    volume = h.volume()  # [N]

    # --- 1. Determine which nodes belong to the upper set -------------------
    # A node is in the upper set iff its volume > size OR it is the root.
    # Additionally it must be reachable from the root via upper-only parents.
    # We BFS from the root, placing nodes in `upper` until we cross the
    # threshold. A child with volume <= size becomes an SPT root.
    upper_set: List[int] = []
    spt_roots_ordered: List[int] = []
    # For each upper node, remember its two "edges" to its children (either
    # upper-idx or spt-idx or -1).
    upper_left_node: List[int] = []  # hierarchy id of left child if upper, else -1
    upper_right_node: List[int] = []
    left_spt_root_node: List[int] = []  # hierarchy id if left is an SPT root, else -1
    right_spt_root_node: List[int] = []

    # Edge case: the root itself has volume <= size. Then the entire scene is
    # a single SPT. We still keep the root in the upper set (with no left/right)
    # and promote its subtree as spt 0.
    root = h.root
    if h.is_leaf[root]:
        # Degenerate: single-node hierarchy.
        upper_set.append(root)
        upper_left_node.append(-1)
        upper_right_node.append(-1)
        left_spt_root_node.append(-1)
        right_spt_root_node.append(-1)
    else:
        queue: List[int] = [root]
        while queue:
            n = queue.pop()
            upper_set.append(n)
            l = int(h.left[n].item())
            r = int(h.right[n].item())

            def process_child(c: int):
                if c < 0:
                    return "none", -1
                if h.is_leaf[c] or volume[c].item() <= size:
                    # Becomes an SPT root (even if it's a leaf — a 1-entry SPT).
                    spt_roots_ordered.append(c)
                    return "spt", c
                queue.append(c)
                return "upper", c

            lk, lid = process_child(l)
            rk, rid = process_child(r)
            upper_left_node.append(lid if lk == "upper" else -1)
            upper_right_node.append(rid if rk == "upper" else -1)
            left_spt_root_node.append(lid if lk == "spt" else -1)
            right_spt_root_node.append(rid if rk == "spt" else -1)

    # --- 2. Build node -> upper_idx map ------------------------------------
    upper_ids = torch.tensor(upper_set, dtype=torch.long)
    node_to_upper = torch.full((N,), -1, dtype=torch.long)
    node_to_upper[upper_ids] = torch.arange(upper_ids.numel(), dtype=torch.long)

    # --- 3. Resolve upper child references to upper indices ---------------
    def to_upper_idx(nid: int) -> int:
        return -1 if nid < 0 else int(node_to_upper[nid].item())

    upper_left = torch.tensor([to_upper_idx(x) for x in upper_left_node], dtype=torch.long)
    upper_right = torch.tensor([to_upper_idx(x) for x in upper_right_node], dtype=torch.long)

    # --- 4. Build SPT id map + spt_root_node_id ----------------------------
    # Preserve insertion order of SPT roots (BFS order from the upper side).
    spt_root_node_id = torch.tensor(spt_roots_ordered, dtype=torch.long)
    node_to_spt = torch.full((N,), -1, dtype=torch.long)
    node_to_spt[spt_root_node_id] = torch.arange(spt_root_node_id.numel(), dtype=torch.long)

    upper_left_spt = torch.tensor(
        [(-1 if x < 0 else int(node_to_spt[x].item())) for x in left_spt_root_node],
        dtype=torch.long,
    )
    upper_right_spt = torch.tensor(
        [(-1 if x < 0 else int(node_to_spt[x].item())) for x in right_spt_root_node],
        dtype=torch.long,
    )

    # --- 5. Build CSR SPT entry arrays -------------------------------------
    # For each SPT, collect its subtree and sort by size(parent(i)) descending
    # (the T-free part of m_d). At cut time m_d(parent) = T * size(parent).
    size_all = h.compute_m_d(1.0, refined=refined_m_d)  # [N]; T=1 gives pure size

    all_entries_node_id: List[int] = []
    all_entries_size_parent: List[float] = []
    offsets: List[int] = [0]

    for r in spt_roots_ordered:
        subtree = _collect_subtree(h, r)  # include r
        # Parent id per entry (root's parent is its hierarchy parent; others
        # use their hierarchy parent, which lies inside the SPT subtree).
        parents = [
            int(h.parent[n].item()) if n != r else int(h.parent[r].item())
            for n in subtree
        ]
        size_parent = [
            float(size_all[p].item()) if p >= 0 else float("inf") for p in parents
        ]
        # Sort descending by size_parent.
        order = sorted(range(len(subtree)), key=lambda i: -size_parent[i])
        for i in order:
            all_entries_node_id.append(subtree[i])
            all_entries_size_parent.append(size_parent[i])
        offsets.append(len(all_entries_node_id))

    spt_offset = torch.tensor(offsets, dtype=torch.long)
    spt_entries_node_id = torch.tensor(all_entries_node_id, dtype=torch.long)
    spt_entries_size_parent = torch.tensor(all_entries_size_parent, dtype=torch.float32)
    # Per-entry SPT membership for the vectorised cut.
    if spt_entries_node_id.numel() > 0:
        # Each SPT s contributes offsets[s+1]-offsets[s] entries.
        lengths = spt_offset[1:] - spt_offset[:-1]  # [S]
        spt_entries_spt_id = torch.repeat_interleave(
            torch.arange(spt_offset.numel() - 1, dtype=torch.long), lengths
        )
    else:
        spt_entries_spt_id = torch.empty(0, dtype=torch.long)

    # --- 6. upper_is_root flag --------------------------------------------
    upper_is_root = torch.zeros(upper_ids.numel(), dtype=torch.bool)
    upper_is_root[0] = True  # BFS from root places root at index 0.

    return HSPT(
        hierarchy=h,
        upper_ids=upper_ids,
        upper_is_root=upper_is_root,
        upper_left=upper_left,
        upper_right=upper_right,
        upper_left_spt=upper_left_spt,
        upper_right_spt=upper_right_spt,
        node_to_upper=node_to_upper,
        spt_root_node_id=spt_root_node_id,
        spt_offset=spt_offset,
        spt_entries_node_id=spt_entries_node_id,
        spt_entries_size_parent=spt_entries_size_parent,
        spt_entries_spt_id=spt_entries_spt_id,
        size_threshold=size,
    )
