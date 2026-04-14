// SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Backward kernel for gsplat.mps.quat_scale_to_covar_preci. Mirrors the
// CUDA implementation in gsplat/cuda/csrc/QuatScaleToCovarCUDA.cu (bwd) and
// the helpers in gsplat/cuda/include/Utils.cuh::quat_scale_to_covar_vjp /
// quat_scale_to_preci_vjp.
//
// Launch: 1-D grid over N Gaussians, one thread per Gaussian. Each thread
// owns its row of v_quats / v_scales, so plain stores; no atomics.
//
// Convention: float3x3 is column-major. M[col][row] addresses the column,
// then the scalar row, matching GLM and the CUDA reference.

#include <metal_stdlib>

using namespace metal;

// Forward declarations for helpers in common.metal (concatenated after this
// translation unit by build.py's ASCII-sorted bundling).
inline float3x3 gsplat_quat_to_rotmat(float4 quat);
inline void gsplat_quat_to_rotmat_vjp(float4 quat,
                                      float3x3 v_R,
                                      thread float4& v_quat);

// Build a symmetric float3x3 from upper-triangle packing
// [m00, m01, m02, m11, m12, m22], halving off-diagonals as the CUDA code
// does — this accounts for v_C being applied to a symmetric forward output.
static inline float3x3 unpack_triu_grad(device const float* g, uint base) {
    float m00 = g[base + 0u];
    float m01 = g[base + 1u];
    float m02 = g[base + 2u];
    float m11 = g[base + 3u];
    float m12 = g[base + 4u];
    float m22 = g[base + 5u];
    float h01 = 0.5f * m01;
    float h02 = 0.5f * m02;
    float h12 = 0.5f * m12;
    return float3x3(
        float3(m00, h01, h02),   // col 0
        float3(h01, m11, h12),   // col 1
        float3(h02, h12, m22)    // col 2
    );
}

// Build float3x3 from a dense row-major [9] block. Output is column-major.
static inline float3x3 unpack_dense_grad(device const float* g, uint base) {
    float r0c0 = g[base + 0u], r0c1 = g[base + 1u], r0c2 = g[base + 2u];
    float r1c0 = g[base + 3u], r1c1 = g[base + 4u], r1c2 = g[base + 5u];
    float r2c0 = g[base + 6u], r2c1 = g[base + 7u], r2c2 = g[base + 8u];
    return float3x3(
        float3(r0c0, r1c0, r2c0),  // col 0
        float3(r0c1, r1c1, r2c1),  // col 1
        float3(r0c2, r1c2, r2c2)   // col 2
    );
}

kernel void gsplat_quat_scale_to_covar_preci_bwd(
    device const float* quats        [[buffer(0)]],  // [N, 4]
    device const float* scales       [[buffer(1)]],  // [N, 3]
    device const float* v_covars     [[buffer(2)]],  // [N,{9|6}] or sentinel
    device const float* v_precis     [[buffer(3)]],  // [N,{9|6}] or sentinel
    device float*       v_quats      [[buffer(4)]],  // [N, 4]
    device float*       v_scales     [[buffer(5)]],  // [N, 3]
    constant uint&      N            [[buffer(6)]],
    constant uint&      triu         [[buffer(7)]],
    constant uint&      has_v_covar  [[buffer(8)]],
    constant uint&      has_v_preci  [[buffer(9)]],
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

    float4 v_quat  = float4(0.0f);
    float3 v_scale = float3(0.0f);

    const uint pack_stride = (triu != 0u) ? 6u : 9u;
    const uint pack_base   = idx * pack_stride;

    if (has_v_covar != 0u) {
        float3x3 v_C = (triu != 0u)
            ? unpack_triu_grad(v_covars, pack_base)
            : unpack_dense_grad(v_covars, pack_base);

        // M = R * diag(scale).
        float3x3 M = float3x3(R[0] * scale.x, R[1] * scale.y, R[2] * scale.z);
        // v_M = (v_C + v_C^T) * M ; v_R = v_M * diag(scale).
        float3x3 v_M = (v_C + transpose(v_C)) * M;
        float3x3 v_R = float3x3(v_M[0] * scale.x,
                                v_M[1] * scale.y,
                                v_M[2] * scale.z);

        gsplat_quat_to_rotmat_vjp(quat, v_R, v_quat);

        // v_scale[i] += dot(R[col=i], v_M[col=i])
        v_scale.x += dot(R[0], v_M[0]);
        v_scale.y += dot(R[1], v_M[1]);
        v_scale.z += dot(R[2], v_M[2]);
    }

    if (has_v_preci != 0u) {
        float3x3 v_P = (triu != 0u)
            ? unpack_triu_grad(v_precis, pack_base)
            : unpack_dense_grad(v_precis, pack_base);

        float3 inv_scale = 1.0f / scale;
        float3x3 M = float3x3(R[0] * inv_scale.x,
                              R[1] * inv_scale.y,
                              R[2] * inv_scale.z);
        float3x3 v_M = (v_P + transpose(v_P)) * M;
        float3x3 v_R = float3x3(v_M[0] * inv_scale.x,
                                v_M[1] * inv_scale.y,
                                v_M[2] * inv_scale.z);

        gsplat_quat_to_rotmat_vjp(quat, v_R, v_quat);

        // d/d(scale[i]) of (1/scale[i]) is -1/scale[i]^2 = -inv_scale[i]^2.
        v_scale.x += -inv_scale.x * inv_scale.x * dot(R[0], v_M[0]);
        v_scale.y += -inv_scale.y * inv_scale.y * dot(R[1], v_M[1]);
        v_scale.z += -inv_scale.z * inv_scale.z * dot(R[2], v_M[2]);
    }

    v_quats[qoff + 0u] = v_quat.x;
    v_quats[qoff + 1u] = v_quat.y;
    v_quats[qoff + 2u] = v_quat.z;
    v_quats[qoff + 3u] = v_quat.w;
    v_scales[soff + 0u] = v_scale.x;
    v_scales[soff + 1u] = v_scale.y;
    v_scales[soff + 2u] = v_scale.z;
}
