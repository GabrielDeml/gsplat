/*
 * SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Config.h"

#if GSPLAT_BUILD_LOD

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>

namespace gsplat {

// -----------------------------------------------------------------------------
// Sphere in frustum: one thread per sphere.
// -----------------------------------------------------------------------------

template <typename scalar_t>
__global__ void lod_sphere_in_frustum_kernel(
    const uint32_t M,
    const scalar_t *__restrict__ mu,     // [M, 3]
    const scalar_t *__restrict__ radius, // [M]
    const scalar_t *__restrict__ planes, // [6, 4]
    bool *__restrict__ out               // [M]
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    const scalar_t mx = mu[3 * i + 0];
    const scalar_t my = mu[3 * i + 1];
    const scalar_t mz = mu[3 * i + 2];
    const scalar_t r = radius[i];

    // For each of the 6 planes (nx, ny, nz, d), signed distance is
    // n.dot(p) + d. Sphere is outside the frustum iff any plane gives
    // dist < -r.
    bool inside = true;
#pragma unroll
    for (int p = 0; p < 6; ++p) {
        const scalar_t nx = planes[4 * p + 0];
        const scalar_t ny = planes[4 * p + 1];
        const scalar_t nz = planes[4 * p + 2];
        const scalar_t d = planes[4 * p + 3];
        const scalar_t dist = nx * mx + ny * my + nz * mz + d;
        if (dist < -r) {
            inside = false;
            break;
        }
    }
    out[i] = inside;
}

void launch_lod_sphere_in_frustum_kernel(
    const at::Tensor &mu,
    const at::Tensor &radius,
    const at::Tensor &planes,
    at::Tensor &out
) {
    const uint32_t M = mu.size(0);
    if (M == 0) return;

    dim3 threads(256);
    dim3 grid((M + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(
        mu.scalar_type(), "lod_sphere_in_frustum_kernel", [&]() {
            lod_sphere_in_frustum_kernel<scalar_t>
                <<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                    M,
                    mu.data_ptr<scalar_t>(),
                    radius.data_ptr<scalar_t>(),
                    planes.data_ptr<scalar_t>(),
                    out.data_ptr<bool>()
                );
        });
}

// -----------------------------------------------------------------------------
// SPT cut count: one thread per touched SPT, binary search on the entries.
// Each SPT's entries are sorted DESCENDING by size_parent. We want the count
// of entries whose size_parent > distance / T.
// -----------------------------------------------------------------------------

__global__ void lod_spt_cut_count_kernel(
    const uint32_t K,
    const float *__restrict__ entries_size_parent,
    const int64_t *__restrict__ offsets,
    const int64_t *__restrict__ touched_spt_ids,
    const float *__restrict__ distances,
    const float inv_T,
    int64_t *__restrict__ cut_count
) {
    const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    const int64_t spt_id = touched_spt_ids[k];
    const int64_t beg = offsets[spt_id];
    const int64_t end = offsets[spt_id + 1];
    const float threshold = distances[k] * inv_T;

    // Binary search for the first index i in [beg, end) such that
    // entries_size_parent[i] <= threshold. That's the count of kept entries.
    int64_t lo = beg;
    int64_t hi = end;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (entries_size_parent[mid] > threshold) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    cut_count[k] = lo - beg;
}

void launch_lod_spt_cut_count_kernel(
    const at::Tensor &entries_size_parent,
    const at::Tensor &offsets,
    const at::Tensor &touched_spt_ids,
    const at::Tensor &distances,
    double T,
    at::Tensor &cut_count
) {
    const uint32_t K = touched_spt_ids.size(0);
    if (K == 0) return;

    dim3 threads(128);
    dim3 grid((K + threads.x - 1) / threads.x);

    float inv_T = 1.0f / static_cast<float>(T <= 0.0 ? 1e-30 : T);

    lod_spt_cut_count_kernel
        <<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            K,
            entries_size_parent.data_ptr<float>(),
            offsets.data_ptr<int64_t>(),
            touched_spt_ids.data_ptr<int64_t>(),
            distances.data_ptr<float>(),
            inv_T,
            cut_count.data_ptr<int64_t>()
        );
}

} // namespace gsplat

#endif
