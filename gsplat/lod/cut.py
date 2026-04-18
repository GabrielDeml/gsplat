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

"""Per-iteration LoD cut.

Given an HSPT, camera parameters, and LoD threshold T, compute the set of
Gaussians (by hierarchy node id) that participate in rendering this frame.

Algorithm (paper §4.3, our BFS + binary-search variant):

    1. BFS the upper hierarchy from its root:
       - frustum-sphere cull with radius = frustum_radius_mult * max_scale(node)
       - if d = ||mu_node - p_cam|| >= m_d(node), add node to upper_cut (stop)
       - else descend:
           - if child is in upper set, enqueue it
           - if child is an SPT root, record (spt_id, d_spt_root) for phase 2

    2. For each (spt_id, d_spt_root) in phase 1:
       - binary-search the SPT's entries (sorted descending by m_d_parent)
         for the largest prefix with m_d_parent > d_spt_root
       - these entries' node ids are added to the render set.

    3. Concatenate upper_cut ids + all SPT cut ids + skybox ids.

This module is GPU-friendly but implemented in plain PyTorch — the CUDA
kernels (`gsplat/cuda/csrc/lod_cut.cu`) are deferred (M6 in the plan).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .frustum import extract_frustum_planes, spheres_in_frustum
from .hspt import HSPT


@dataclass
class RenderSet:
    """Output of ``compute_render_set``.

    ``node_ids`` indexes the underlying ``GaussianHierarchy``; the accompanying
    ``upper_cut_ids`` / ``spt_cut_ids`` break it down by source for logging and
    for the caching layer.

    ``spt_ids_touched`` / ``spt_distances`` are returned so the caller can
    update a GPU cache: each touched SPT was cut at its recorded distance.
    """

    node_ids: Tensor  # [M] long, all Gaussian hierarchy ids to render
    upper_cut_ids: Tensor  # [U'] long, subset of node_ids from the upper cut
    spt_cut_ids: Tensor  # [S'] long, subset of node_ids from SPT cuts
    spt_ids_touched: Tensor  # [T] long, SPT ids whose cuts contributed
    spt_distances: Tensor  # [T] float, camera distances used for those cuts


@torch.no_grad()
def compute_render_set(
    hspt: HSPT,
    camera_pos: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    T: float,
    frustum_radius_mult: float = 3.0,
    near: float = 0.01,
    far: float = 1e6,
    skip_frustum: bool = False,
    refined_m_d: bool = False,
    force_cpu: bool = False,
) -> RenderSet:
    """Compute the render set for one camera view.

    Args:
        hspt: the HSPT over a ``GaussianHierarchy``.
        camera_pos: [3] world-space camera centre.
        viewmat: [4, 4] world-to-camera matrix.
        K: [3, 3] intrinsics.
        width, height: pixel dimensions (for frustum planes).
        T: LoD threshold (m_d = T * size-like quantity).
        frustum_radius_mult: radius = mult * max_scale(node) (paper uses 3).
        skip_frustum: disable frustum culling (useful in tests).
        refined_m_d: use Eq. 6 (sqrt of surface-area-like quantity) instead of
            Eq. 3 (max scale component).

    Returns:
        ``RenderSet``. All returned tensors live on CPU (caller uploads the
        subset to GPU as needed for rendering).
    """
    # Dispatch to the GPU-parallel implementation when the HSPT metadata is
    # on a non-CPU device. The Python path below is the canonical CPU
    # fallback (also exercised by tests regardless of hardware).
    if not force_cpu and hspt.upper_ids.device.type != "cpu":
        from .cut_gpu import compute_render_set_gpu

        return compute_render_set_gpu(
            hspt,
            camera_pos,
            viewmat,
            K,
            width,
            height,
            T,
            frustum_radius_mult=frustum_radius_mult,
            near=near,
            far=far,
            skip_frustum=skip_frustum,
            refined_m_d=refined_m_d,
        )

    h = hspt.hierarchy
    device = camera_pos.device
    camera_pos_cpu = camera_pos.detach().to("cpu").view(3)

    # --- Precompute ------------------------------------------------------
    # Compute m_d and max_scale on all upper nodes once (cached on the HSPT
    # instance until T changes).
    if (
        hspt.upper_m_d is None
        or hspt.upper_max_scale is None
        or hspt.upper_mu is None
    ):
        hspt.precompute_upper_metrics(T, refined=refined_m_d)

    upper_mu = hspt.upper_mu  # [U, 3]
    upper_max_scale = hspt.upper_max_scale  # [U]
    upper_m_d = hspt.upper_m_d  # [U]

    # Frustum planes for this view.
    if skip_frustum:
        planes = None
    else:
        planes = extract_frustum_planes(
            viewmat.detach().to("cpu"),
            K.detach().to("cpu"),
            width,
            height,
            near=near,
            far=far,
        )

    # --- Phase 1: BFS the upper hierarchy --------------------------------
    upper_cut_list: List[int] = []  # hierarchy node ids
    spt_ids_touched: List[int] = []
    spt_distances: List[float] = []

    # We use an explicit stack to avoid heavy queue ops; BFS order isn't
    # required for correctness, only that we traverse top-down.
    stack: List[int] = [0]  # index 0 is the root in the upper array
    while stack:
        u = stack.pop()
        node_id = int(hspt.upper_ids[u].item())
        mu_u = upper_mu[u]
        r_u = float(upper_max_scale[u].item() * frustum_radius_mult)
        # Frustum test.
        if planes is not None:
            # Single-sphere test, fast path
            inside = bool(
                spheres_in_frustum(
                    mu_u.unsqueeze(0),
                    torch.tensor([r_u], dtype=mu_u.dtype),
                    planes,
                ).item()
            )
            if not inside:
                continue
        d = float((mu_u - camera_pos_cpu).norm().item())
        m_d_u = float(upper_m_d[u].item())

        if d >= m_d_u:
            # Coarse enough; add to cut, do not descend.
            upper_cut_list.append(node_id)
            continue

        # Descend.
        ul = int(hspt.upper_left[u].item())
        ur = int(hspt.upper_right[u].item())
        ls = int(hspt.upper_left_spt[u].item())
        rs = int(hspt.upper_right_spt[u].item())

        if ul >= 0:
            stack.append(ul)
        elif ls >= 0:
            # Frustum cull the SPT root too.
            spt_root_id = int(hspt.spt_root_node_id[ls].item())
            mu_spt = h.mu[spt_root_id]
            r_spt = float(h.max_scale(torch.tensor([spt_root_id])).item() * frustum_radius_mult)
            if planes is not None:
                keep = bool(
                    spheres_in_frustum(
                        mu_spt.unsqueeze(0),
                        torch.tensor([r_spt], dtype=mu_spt.dtype),
                        planes,
                    ).item()
                )
            else:
                keep = True
            if keep:
                d_spt = float((mu_spt - camera_pos_cpu).norm().item())
                spt_ids_touched.append(ls)
                spt_distances.append(d_spt)

        if ur >= 0:
            stack.append(ur)
        elif rs >= 0:
            spt_root_id = int(hspt.spt_root_node_id[rs].item())
            mu_spt = h.mu[spt_root_id]
            r_spt = float(h.max_scale(torch.tensor([spt_root_id])).item() * frustum_radius_mult)
            if planes is not None:
                keep = bool(
                    spheres_in_frustum(
                        mu_spt.unsqueeze(0),
                        torch.tensor([r_spt], dtype=mu_spt.dtype),
                        planes,
                    ).item()
                )
            else:
                keep = True
            if keep:
                d_spt = float((mu_spt - camera_pos_cpu).norm().item())
                spt_ids_touched.append(rs)
                spt_distances.append(d_spt)

    # --- Phase 2: per-SPT binary-search cut ------------------------------
    # Entries store size_parent (= m_d_parent / T). The cut condition
    # "m_d(parent) > d" becomes "size_parent > d / T". Sort order is
    # descending in size_parent, so we find the largest prefix satisfying it.
    spt_cut_ids_list: List[Tensor] = []
    for spt_id, d_spt in zip(spt_ids_touched, spt_distances):
        beg, end = hspt.spt_range(spt_id)
        size_parent = hspt.spt_entries_size_parent[beg:end]
        threshold = d_spt / max(T, 1e-30)
        # size_parent is sorted DESCENDING. We want count of entries where
        # size_parent > threshold. Equivalently: negate to ascending and use
        # searchsorted to find the insertion point for -threshold.
        neg = -size_parent
        k = int(torch.searchsorted(neg, torch.tensor(-threshold)).item())
        if k > 0:
            spt_cut_ids_list.append(hspt.spt_entries_node_id[beg : beg + k])

    upper_cut_ids = torch.tensor(upper_cut_list, dtype=torch.long)
    spt_cut_ids = (
        torch.cat(spt_cut_ids_list) if spt_cut_ids_list else torch.empty(0, dtype=torch.long)
    )
    node_ids = torch.cat([upper_cut_ids, spt_cut_ids])

    return RenderSet(
        node_ids=node_ids,
        upper_cut_ids=upper_cut_ids,
        spt_cut_ids=spt_cut_ids,
        spt_ids_touched=torch.tensor(spt_ids_touched, dtype=torch.long),
        spt_distances=torch.tensor(spt_distances, dtype=torch.float32),
    )
