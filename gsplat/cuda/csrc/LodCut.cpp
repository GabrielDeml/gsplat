/*
 * SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Config.h"

#if GSPLAT_BUILD_LOD

#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h>

#include "Common.h"
#include "LodCut.h"

namespace gsplat {

at::Tensor lod_sphere_in_frustum(
    const at::Tensor &mu,      // [M, 3]
    const at::Tensor &radius,  // [M]
    const at::Tensor &planes   // [6, 4]
) {
    DEVICE_GUARD(mu);
    TORCH_CHECK(mu.dim() == 2 && mu.size(-1) == 3, "mu: [M, 3]");
    TORCH_CHECK(radius.dim() == 1 && radius.size(0) == mu.size(0),
                "radius: [M] matching mu");
    TORCH_CHECK(planes.dim() == 2 && planes.size(0) == 6 && planes.size(1) == 4,
                "planes: [6, 4]");
    auto out = at::empty({mu.size(0)}, mu.options().dtype(at::kBool));
    launch_lod_sphere_in_frustum_kernel(mu, radius, planes, out);
    return out;
}

at::Tensor lod_spt_cut_count(
    const at::Tensor &entries_size_parent, // [E] sorted desc per SPT
    const at::Tensor &offsets,             // [S+1]
    const at::Tensor &touched_spt_ids,     // [K]
    const at::Tensor &distances,           // [K]
    double T
) {
    DEVICE_GUARD(entries_size_parent);
    TORCH_CHECK(entries_size_parent.dim() == 1);
    TORCH_CHECK(offsets.dim() == 1);
    TORCH_CHECK(touched_spt_ids.dim() == 1);
    TORCH_CHECK(distances.dim() == 1 && distances.size(0) == touched_spt_ids.size(0));
    auto cut_count = at::empty(
        {touched_spt_ids.size(0)},
        touched_spt_ids.options()
    );
    launch_lod_spt_cut_count_kernel(
        entries_size_parent, offsets, touched_spt_ids, distances, T, cut_count
    );
    return cut_count;
}

} // namespace gsplat

#endif
