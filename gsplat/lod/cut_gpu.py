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

"""GPU-parallel LoD cut.

Mirror of ``gsplat/lod/cut.py::compute_render_set`` but designed to run with
every per-frame computation on-device:

    - Level-synchronous BFS over the upper hierarchy: all nodes at a given
      depth are evaluated in one tensor op (frustum test + m_d vs distance),
      and the next level is gathered with ``torch.index_select``.
    - Single vectorised SPT-cut step: for every entry across every touched
      SPT, compare its ``size_parent`` against that SPT's threshold in one
      ``>`` op. Because entries are sorted descending per SPT, the kept mask
      is a prefix per group — equivalent to the per-SPT binary search.
    - Frustum-sphere test is vectorised over batches (already the case).

When the associated CUDA build flag is enabled the raw kernel in
``gsplat/cuda/csrc/LodCutCUDA.cu`` is used instead of PyTorch primitives for
the frustum test. That path is dispatched through
``gsplat.cuda._wrapper.lod_sphere_in_frustum`` and falls back to PyTorch
automatically when the op is not registered (e.g. a non-CUDA install).
"""

from typing import Optional

import torch
from torch import Tensor

from .cut import RenderSet
from .frustum import extract_frustum_planes, spheres_in_frustum
from .hspt import HSPT


def _sphere_in_frustum_maybe_cuda(
    mu: Tensor, radius: Tensor, planes: Tensor
) -> Tensor:
    """Route through the raw CUDA kernel if available; else PyTorch."""
    if mu.device.type == "cuda":
        try:
            from ..cuda._wrapper import lod_sphere_in_frustum  # type: ignore

            return lod_sphere_in_frustum(mu, radius, planes)
        except (ImportError, AttributeError):
            pass
    return spheres_in_frustum(mu, radius, planes)


@torch.no_grad()
def compute_render_set_gpu(
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
) -> RenderSet:
    """GPU-parallel LoD cut.

    Expects the ``hspt`` metadata tensors to already live on the same device
    as ``camera_pos`` / ``viewmat`` (use ``hspt.to(device)`` once after
    building / rebuilding the HSPT).
    """
    device = camera_pos.device
    dtype = hspt.upper_ids.dtype  # long

    # --- Precompute upper aggregates on-device, if not already -----------
    if hspt.upper_m_d is None or hspt.upper_m_d.device != device:
        hspt.precompute_upper_metrics(T, refined=refined_m_d)
        if hspt.upper_mu.device != device:
            hspt.upper_mu = hspt.upper_mu.to(device)
            hspt.upper_max_scale = hspt.upper_max_scale.to(device)
            hspt.upper_m_d = hspt.upper_m_d.to(device)

    upper_mu = hspt.upper_mu
    upper_max_scale = hspt.upper_max_scale
    upper_m_d = hspt.upper_m_d

    # Frustum planes (trivial cost; compute once).
    if skip_frustum:
        planes = None
    else:
        planes = extract_frustum_planes(
            viewmat.to(device).float(), K.to(device).float(), width, height, near=near, far=far
        )

    # --- Level-synchronous BFS ------------------------------------------
    U = hspt.upper_ids.numel()
    upper_left = hspt.upper_left
    upper_right = hspt.upper_right
    upper_left_spt = hspt.upper_left_spt
    upper_right_spt = hspt.upper_right_spt

    # Start at the upper root (upper index 0).
    if U == 0:
        return RenderSet(
            node_ids=torch.empty(0, dtype=dtype, device=device),
            upper_cut_ids=torch.empty(0, dtype=dtype, device=device),
            spt_cut_ids=torch.empty(0, dtype=dtype, device=device),
            spt_ids_touched=torch.empty(0, dtype=dtype, device=device),
            spt_distances=torch.empty(0, dtype=torch.float32, device=device),
        )

    upper_cut_list = []
    spt_touched_list = []
    spt_dist_list = []

    frontier = torch.tensor([0], dtype=torch.long, device=device)  # upper indices

    while frontier.numel() > 0:
        mu = upper_mu[frontier]  # [F, 3]
        r = upper_max_scale[frontier] * frustum_radius_mult  # [F]
        if planes is not None:
            visible = _sphere_in_frustum_maybe_cuda(mu, r, planes)  # [F] bool
        else:
            visible = torch.ones(frontier.numel(), dtype=torch.bool, device=device)

        d = (mu - camera_pos.to(device).view(1, 3)).norm(dim=-1)  # [F]
        m_d = upper_m_d[frontier]  # [F]
        coarse_enough = (d >= m_d) & visible

        # Stop at these upper nodes.
        keep_idx = frontier[coarse_enough]
        if keep_idx.numel() > 0:
            upper_cut_list.append(hspt.upper_ids[keep_idx])

        descend_mask = visible & ~coarse_enough
        descend_idx = frontier[descend_mask]

        if descend_idx.numel() == 0:
            break

        # Gather children: four slots per parent (left/right × upper/spt).
        ul = upper_left[descend_idx]
        ur = upper_right[descend_idx]
        ls = upper_left_spt[descend_idx]
        rs = upper_right_spt[descend_idx]

        # Next frontier = upper-children that are valid.
        upper_children = torch.cat([ul, ur])
        upper_children = upper_children[upper_children >= 0]

        # SPT roots touched: the union of ls/rs where >= 0, frustum-tested.
        spt_candidates = torch.cat([ls, rs])
        spt_candidates = spt_candidates[spt_candidates >= 0]
        if spt_candidates.numel() > 0:
            spt_root_node_ids = hspt.spt_root_node_id[spt_candidates]  # [K]
            spt_mu = hspt.hierarchy.mu[spt_root_node_ids.to("cpu")].to(device)
            spt_scales_exp = torch.exp(
                hspt.hierarchy.scale[spt_root_node_ids.to("cpu")].to(device)
            )
            spt_max_s = spt_scales_exp.amax(dim=-1)
            spt_r = spt_max_s * frustum_radius_mult
            if planes is not None:
                vis = _sphere_in_frustum_maybe_cuda(spt_mu, spt_r, planes)
                spt_candidates = spt_candidates[vis]
                if spt_candidates.numel() > 0:
                    spt_root_node_ids = hspt.spt_root_node_id[spt_candidates]
                    spt_mu = hspt.hierarchy.mu[spt_root_node_ids.to("cpu")].to(device)
            if spt_candidates.numel() > 0:
                d_spt = (spt_mu - camera_pos.to(device).view(1, 3)).norm(dim=-1)
                spt_touched_list.append(spt_candidates)
                spt_dist_list.append(d_spt)

        frontier = upper_children

    if upper_cut_list:
        upper_cut_ids_global = torch.cat(
            [hspt.upper_ids[0].new_empty(0)] + []  # keep dtype
        )
        upper_cut_ids_global = torch.cat(upper_cut_list)
    else:
        upper_cut_ids_global = torch.empty(0, dtype=dtype, device=device)

    # --- Per-SPT vectorised cut -----------------------------------------
    if spt_touched_list:
        spt_ids_touched = torch.cat(spt_touched_list)
        spt_dists = torch.cat(spt_dist_list)
    else:
        spt_ids_touched = torch.empty(0, dtype=dtype, device=device)
        spt_dists = torch.empty(0, dtype=torch.float32, device=device)

    if spt_ids_touched.numel() > 0:
        # Per-SPT threshold: size must be > d / T to be kept.
        # We must also mask the entries to only those belonging to a TOUCHED SPT.
        # Build a per-SPT-id threshold dense vector, with inf for untouched SPTs
        # (so nothing passes the > test).
        S_total = hspt.spt_offset.numel() - 1
        thresholds = torch.full(
            (S_total,), float("inf"), dtype=torch.float32, device=device
        )
        thresholds[spt_ids_touched] = spt_dists / max(T, 1e-30)

        entry_thr = thresholds[hspt.spt_entries_spt_id]  # [E]
        keep_mask = hspt.spt_entries_size_parent > entry_thr  # [E]
        spt_cut_ids = hspt.spt_entries_node_id[keep_mask]
    else:
        spt_cut_ids = torch.empty(0, dtype=dtype, device=device)

    node_ids = torch.cat([upper_cut_ids_global, spt_cut_ids])

    return RenderSet(
        node_ids=node_ids,
        upper_cut_ids=upper_cut_ids_global,
        spt_cut_ids=spt_cut_ids,
        spt_ids_touched=spt_ids_touched,
        spt_distances=spt_dists,
    )
