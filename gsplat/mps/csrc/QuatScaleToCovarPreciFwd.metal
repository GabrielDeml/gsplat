// SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Forward kernel for gsplat.mps.quat_scale_to_covar_preci.
// Ports gsplat/cuda/csrc/QuatScaleToCovarCUDA.cu (fwd) + the helper math in
// gsplat/cuda/include/Utils.cuh::quat_scale_to_covar_preci.
//
// Launch: 1-D grid over N Gaussians, one thread per Gaussian. No
// threadgroup/shared memory; default group_size supplied by the PyTorch
// dispatcher is fine.
//
// Output layout (matches CUDA bit-for-bit):
//   triu=false : dense row-major  [N,3,3] -> N*9 floats per output
//   triu=true  : upper-triangle  [N,6]    -> N*6 floats per output
//                entries = [m(0,0), m(0,1), m(0,2), m(1,1), m(1,2), m(2,2)]
//
// compute_covar / compute_preci toggle stores. When one is disabled the
// Python wrapper passes a size-1 sentinel tensor for the unused buffer; the
// flag then causes the kernel to skip all stores into it.

#include <metal_stdlib>

using namespace metal;

// Forward-declared helpers defined in common.metal. Concatenation order in
// build.py is ASCII by filename, so uppercase filenames like this one land
// BEFORE common.metal in the translation unit.
inline float3x3 gsplat_quat_to_rotmat(float4 quat);

kernel void gsplat_quat_scale_to_covar_preci_fwd(
    device const float* quats         [[buffer(0)]],  // [N, 4]
    device const float* scales        [[buffer(1)]],  // [N, 3]
    device float*       covars_out    [[buffer(2)]],  // [N,{9|6}] or sentinel
    device float*       precis_out    [[buffer(3)]],  // [N,{9|6}] or sentinel
    constant uint&      N             [[buffer(4)]],
    constant uint&      triu          [[buffer(5)]],
    constant uint&      compute_covar [[buffer(6)]],
    constant uint&      compute_preci [[buffer(7)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= N) {
        return;
    }

    const uint qoff = idx * 4u;
    const uint soff = idx * 3u;
    float4 quat  = float4(quats[qoff + 0u], quats[qoff + 1u],
                          quats[qoff + 2u], quats[qoff + 3u]);
    float3 scale = float3(scales[soff + 0u], scales[soff + 1u],
                          scales[soff + 2u]);

    float3x3 R = gsplat_quat_to_rotmat(quat);

    if (compute_covar != 0u) {
        // M = R * diag(scale); C = M * M^T.
        float3x3 M = float3x3(R[0] * scale.x, R[1] * scale.y, R[2] * scale.z);
        float3x3 C = M * transpose(M);

        if (triu != 0u) {
            const uint o = idx * 6u;
            covars_out[o + 0u] = C[0][0];
            covars_out[o + 1u] = C[1][0];
            covars_out[o + 2u] = C[2][0];
            covars_out[o + 3u] = C[1][1];
            covars_out[o + 4u] = C[2][1];
            covars_out[o + 5u] = C[2][2];
        } else {
            const uint o = idx * 9u;
            // Row-major write. Matrix is symmetric so column/row order is
            // irrelevant for correctness, but we mirror CUDA literally.
            covars_out[o + 0u] = C[0][0];
            covars_out[o + 1u] = C[1][0];
            covars_out[o + 2u] = C[2][0];
            covars_out[o + 3u] = C[0][1];
            covars_out[o + 4u] = C[1][1];
            covars_out[o + 5u] = C[2][1];
            covars_out[o + 6u] = C[0][2];
            covars_out[o + 7u] = C[1][2];
            covars_out[o + 8u] = C[2][2];
        }
    }

    if (compute_preci != 0u) {
        // P = R * diag(1/scale); Prec = P * P^T. Scale must be non-zero; the
        // projection pipeline culls degenerate Gaussians upstream.
        float3 inv_scale = 1.0f / scale;
        float3x3 P = float3x3(R[0] * inv_scale.x,
                              R[1] * inv_scale.y,
                              R[2] * inv_scale.z);
        float3x3 Q = P * transpose(P);

        if (triu != 0u) {
            const uint o = idx * 6u;
            precis_out[o + 0u] = Q[0][0];
            precis_out[o + 1u] = Q[1][0];
            precis_out[o + 2u] = Q[2][0];
            precis_out[o + 3u] = Q[1][1];
            precis_out[o + 4u] = Q[2][1];
            precis_out[o + 5u] = Q[2][2];
        } else {
            const uint o = idx * 9u;
            precis_out[o + 0u] = Q[0][0];
            precis_out[o + 1u] = Q[1][0];
            precis_out[o + 2u] = Q[2][0];
            precis_out[o + 3u] = Q[0][1];
            precis_out[o + 4u] = Q[1][1];
            precis_out[o + 5u] = Q[2][1];
            precis_out[o + 6u] = Q[0][2];
            precis_out[o + 7u] = Q[1][2];
            precis_out[o + 8u] = Q[2][2];
        }
    }
}
