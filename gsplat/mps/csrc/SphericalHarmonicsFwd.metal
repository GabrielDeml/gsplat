// SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Forward kernel for gsplat.mps.spherical_harmonics.
// Ports gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu::spherical_harmonics_fwd_kernel
// and the per-channel helper sh_coeffs_to_color_fast (same file, line 37).
//
// Launch: 1-D grid over N Gaussians, one thread per Gaussian. Each thread
// computes all 3 output channels inline. The CUDA kernel parallelises over
// N*3 and therefore needs atomic accumulation for v_dirs in the backward.
// We trade a factor of 3 of launch parallelism for atomic-free writes in
// both directions; at realistic scene sizes (~100k+) the Apple GPU is
// already saturated.
//
// No helpers from common.metal are referenced (no atomics, no quaternion
// math); this file is self-contained.

#include <metal_stdlib>

using namespace metal;

kernel void gsplat_spherical_harmonics_fwd(
    device const float* dirs        [[buffer(0)]],  // [N, 3]
    device const float* coeffs      [[buffer(1)]],  // [N, K, 3] row-major
    device const uchar* masks       [[buffer(2)]],  // [N] (bool) or sentinel
    device float*       colors_out  [[buffer(3)]],  // [N, 3]
    constant uint&      N           [[buffer(4)]],
    constant uint&      K           [[buffer(5)]],
    constant uint&      degrees     [[buffer(6)]],
    constant uint&      has_mask    [[buffer(7)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= N) {
        return;
    }
    if (has_mask != 0u && masks[idx] == 0u) {
        // Output was pre-zeroed by the Python wrapper.
        return;
    }

    const uint coff = idx * K * 3u;
    const uint ooff = idx * 3u;
    const uint doff = idx * 3u;

    // Band 0: constant 0.2820947917738781 * coeffs[0, :].
    const float band0 = 0.2820947917738781f;
    float3 result = float3(band0 * coeffs[coff + 0u],
                           band0 * coeffs[coff + 1u],
                           band0 * coeffs[coff + 2u]);

    if (degrees >= 1u) {
        float dx = dirs[doff + 0u];
        float dy = dirs[doff + 1u];
        float dz = dirs[doff + 2u];
        float inorm = rsqrt(dx * dx + dy * dy + dz * dz);
        float x = dx * inorm;
        float y = dy * inorm;
        float z = dz * inorm;

        // Band 1 bases (applied to coeffs[1..3]).
        const float c1 = 0.48860251190292f;
        float3 c_1 = float3(coeffs[coff + 1u*3u + 0u], coeffs[coff + 1u*3u + 1u], coeffs[coff + 1u*3u + 2u]);
        float3 c_2 = float3(coeffs[coff + 2u*3u + 0u], coeffs[coff + 2u*3u + 1u], coeffs[coff + 2u*3u + 2u]);
        float3 c_3 = float3(coeffs[coff + 3u*3u + 0u], coeffs[coff + 3u*3u + 1u], coeffs[coff + 3u*3u + 2u]);
        result += c1 * (-y * c_1 + z * c_2 - x * c_3);

        if (degrees >= 2u) {
            float z2 = z * z;
            float fTmp0B = -1.092548430592079f * z;
            float fC1 = x * x - y * y;
            float fS1 = 2.0f * x * y;
            float pSH6 = 0.9461746957575601f * z2 - 0.3153915652525201f;
            float pSH7 = fTmp0B * x;
            float pSH5 = fTmp0B * y;
            float pSH8 = 0.5462742152960395f * fC1;
            float pSH4 = 0.5462742152960395f * fS1;

            float3 c_4 = float3(coeffs[coff + 4u*3u + 0u], coeffs[coff + 4u*3u + 1u], coeffs[coff + 4u*3u + 2u]);
            float3 c_5 = float3(coeffs[coff + 5u*3u + 0u], coeffs[coff + 5u*3u + 1u], coeffs[coff + 5u*3u + 2u]);
            float3 c_6 = float3(coeffs[coff + 6u*3u + 0u], coeffs[coff + 6u*3u + 1u], coeffs[coff + 6u*3u + 2u]);
            float3 c_7 = float3(coeffs[coff + 7u*3u + 0u], coeffs[coff + 7u*3u + 1u], coeffs[coff + 7u*3u + 2u]);
            float3 c_8 = float3(coeffs[coff + 8u*3u + 0u], coeffs[coff + 8u*3u + 1u], coeffs[coff + 8u*3u + 2u]);

            result += pSH4 * c_4 + pSH5 * c_5 + pSH6 * c_6 + pSH7 * c_7 + pSH8 * c_8;

            if (degrees >= 3u) {
                float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
                float fTmp1B = 1.445305721320277f * z;
                float fC2 = x * fC1 - y * fS1;
                float fS2 = x * fS1 + y * fC1;
                float pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
                float pSH13 = fTmp0C * x;
                float pSH11 = fTmp0C * y;
                float pSH14 = fTmp1B * fC1;
                float pSH10 = fTmp1B * fS1;
                float pSH15 = -0.5900435899266435f * fC2;
                float pSH9  = -0.5900435899266435f * fS2;

                float3 c_9  = float3(coeffs[coff +  9u*3u + 0u], coeffs[coff +  9u*3u + 1u], coeffs[coff +  9u*3u + 2u]);
                float3 c_10 = float3(coeffs[coff + 10u*3u + 0u], coeffs[coff + 10u*3u + 1u], coeffs[coff + 10u*3u + 2u]);
                float3 c_11 = float3(coeffs[coff + 11u*3u + 0u], coeffs[coff + 11u*3u + 1u], coeffs[coff + 11u*3u + 2u]);
                float3 c_12 = float3(coeffs[coff + 12u*3u + 0u], coeffs[coff + 12u*3u + 1u], coeffs[coff + 12u*3u + 2u]);
                float3 c_13 = float3(coeffs[coff + 13u*3u + 0u], coeffs[coff + 13u*3u + 1u], coeffs[coff + 13u*3u + 2u]);
                float3 c_14 = float3(coeffs[coff + 14u*3u + 0u], coeffs[coff + 14u*3u + 1u], coeffs[coff + 14u*3u + 2u]);
                float3 c_15 = float3(coeffs[coff + 15u*3u + 0u], coeffs[coff + 15u*3u + 1u], coeffs[coff + 15u*3u + 2u]);

                result += pSH9 * c_9 + pSH10 * c_10 + pSH11 * c_11 + pSH12 * c_12
                        + pSH13 * c_13 + pSH14 * c_14 + pSH15 * c_15;

                if (degrees >= 4u) {
                    float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
                    float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
                    float fTmp2B = -1.770130769779931f * z;
                    float fC3 = x * fC2 - y * fS2;
                    float fS3 = x * fS2 + y * fC2;
                    float pSH20 = 1.984313483298443f * z * pSH12 - 1.006230589874905f * pSH6;
                    float pSH21 = fTmp0D * x;
                    float pSH19 = fTmp0D * y;
                    float pSH22 = fTmp1C * fC1;
                    float pSH18 = fTmp1C * fS1;
                    float pSH23 = fTmp2B * fC2;
                    float pSH17 = fTmp2B * fS2;
                    float pSH24 = 0.6258357354491763f * fC3;
                    float pSH16 = 0.6258357354491763f * fS3;

                    float3 c_16 = float3(coeffs[coff + 16u*3u + 0u], coeffs[coff + 16u*3u + 1u], coeffs[coff + 16u*3u + 2u]);
                    float3 c_17 = float3(coeffs[coff + 17u*3u + 0u], coeffs[coff + 17u*3u + 1u], coeffs[coff + 17u*3u + 2u]);
                    float3 c_18 = float3(coeffs[coff + 18u*3u + 0u], coeffs[coff + 18u*3u + 1u], coeffs[coff + 18u*3u + 2u]);
                    float3 c_19 = float3(coeffs[coff + 19u*3u + 0u], coeffs[coff + 19u*3u + 1u], coeffs[coff + 19u*3u + 2u]);
                    float3 c_20 = float3(coeffs[coff + 20u*3u + 0u], coeffs[coff + 20u*3u + 1u], coeffs[coff + 20u*3u + 2u]);
                    float3 c_21 = float3(coeffs[coff + 21u*3u + 0u], coeffs[coff + 21u*3u + 1u], coeffs[coff + 21u*3u + 2u]);
                    float3 c_22 = float3(coeffs[coff + 22u*3u + 0u], coeffs[coff + 22u*3u + 1u], coeffs[coff + 22u*3u + 2u]);
                    float3 c_23 = float3(coeffs[coff + 23u*3u + 0u], coeffs[coff + 23u*3u + 1u], coeffs[coff + 23u*3u + 2u]);
                    float3 c_24 = float3(coeffs[coff + 24u*3u + 0u], coeffs[coff + 24u*3u + 1u], coeffs[coff + 24u*3u + 2u]);

                    result += pSH16 * c_16 + pSH17 * c_17 + pSH18 * c_18 + pSH19 * c_19
                            + pSH20 * c_20 + pSH21 * c_21 + pSH22 * c_22 + pSH23 * c_23
                            + pSH24 * c_24;
                }
            }
        }
    }

    colors_out[ooff + 0u] = result.x;
    colors_out[ooff + 1u] = result.y;
    colors_out[ooff + 2u] = result.z;
}
