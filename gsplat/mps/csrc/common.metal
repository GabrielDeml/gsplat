// SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Shared device-side utilities for gsplat MPS kernels.
//
// Concatenation note: gsplat/mps/build.py sorts .metal files by path in
// ASCII order, so files whose names start with an uppercase letter
// (e.g. RasterizeToPixels3DGSFwd.metal) are emitted BEFORE this file in
// the single compile unit. Tier-1 kernel files that call helpers from
// this file must include a forward-declaration block for the symbols
// they use.

#include <metal_stdlib>
#include <metal_atomic>

using namespace metal;

// Tile dimensions used by the 3DGS/2DGS rasterizers. Matches the CUDA
// default grid layout (block.thread_index().{x,y} up to 16).
constant constexpr uint GSPLAT_TILE_WIDTH  = 16;
constant constexpr uint GSPLAT_TILE_HEIGHT = 16;

// Quaternion -> rotation matrix. Convention (w, x, y, z) packed as
// (q.x, q.y, q.z, q.w) — matches gsplat/cuda/include/Utils.cuh and the
// PyTorch oracle in gsplat/cuda/_math.py::_quat_to_rotmat.
// Returned matrix is column-major, bit-compatible with the GLM mat3
// produced by the CUDA helper.
inline float3x3 gsplat_quat_to_rotmat(float4 quat) {
    float w = quat.x;
    float x = quat.y;
    float y = quat.z;
    float z = quat.w;

    float inv_norm = rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;

    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;

    float3 col0 = float3(1.0f - 2.0f * (y2 + z2),
                         2.0f * (xy + wz),
                         2.0f * (xz - wy));
    float3 col1 = float3(2.0f * (xy - wz),
                         1.0f - 2.0f * (x2 + z2),
                         2.0f * (yz + wx));
    float3 col2 = float3(2.0f * (xz + wy),
                         2.0f * (yz - wx),
                         1.0f - 2.0f * (x2 + y2));
    return float3x3(col0, col1, col2);
}

// Closed-form 3x3 inverse via cofactor/adjugate. Caller is responsible
// for ensuring the input is non-singular; we do not guard against
// det(m) == 0.
inline float3x3 gsplat_mat3_inverse(float3x3 m) {
    float3 c0 = m[0];
    float3 c1 = m[1];
    float3 c2 = m[2];

    float3 r0 = float3(c1.y * c2.z - c1.z * c2.y,
                       c0.z * c2.y - c0.y * c2.z,
                       c0.y * c1.z - c0.z * c1.y);
    float3 r1 = float3(c1.z * c2.x - c1.x * c2.z,
                       c0.x * c2.z - c0.z * c2.x,
                       c0.z * c1.x - c0.x * c1.z);
    float3 r2 = float3(c1.x * c2.y - c1.y * c2.x,
                       c0.y * c2.x - c0.x * c2.y,
                       c0.x * c1.y - c0.y * c1.x);

    float det = c0.x * r0.x + c0.y * r1.x + c0.z * r2.x;
    float inv_det = 1.0f / det;

    // Adjugate is the transpose of the cofactor matrix; the rows above
    // already correspond to columns of the inverse.
    return float3x3(r0 * inv_det, r1 * inv_det, r2 * inv_det);
}

// Closed-form 2x2 inverse. Caller must ensure det(m) != 0.
inline float2x2 gsplat_mat2_inverse(float2x2 m) {
    float a = m[0].x, b = m[1].x;
    float c = m[0].y, d = m[1].y;
    float inv_det = 1.0f / (a * d - b * c);
    float2 col0 = float2( d, -c) * inv_det;
    float2 col1 = float2(-b,  a) * inv_det;
    return float2x2(col0, col1);
}

// Relaxed atomic float add. Metal 3+ provides atomic<float> with
// fetch-add as an explicit-order operation.
inline void gsplat_atomic_add_float(device atomic<float>* addr, float val) {
    atomic_fetch_add_explicit(addr, val, memory_order_relaxed);
}
