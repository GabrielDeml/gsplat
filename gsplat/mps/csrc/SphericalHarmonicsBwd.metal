// SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Backward kernel for gsplat.mps.spherical_harmonics.
// Ports gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu::spherical_harmonics_bwd_kernel
// and the per-channel VJP sh_coeffs_to_color_fast_vjp (same file, line 129).
//
// Launch: 1-D grid over N Gaussians, one thread per Gaussian. Each thread
// owns its row of v_coeffs and v_dirs outright, so all writes are plain
// stores — no atomics. The CUDA implementation parallelises over N*3 and
// therefore uses gpuAtomicAdd for v_dirs; we vectorise over the 3 output
// channels with float3 ops instead.
//
// Mask semantics: when has_mask is set and masks[idx]==0, v_coeffs and
// v_dirs rows for this index are left at their pre-zeroed values.
//
// compute_v_dirs=0 disables the v_dirs path entirely so the Python wrapper
// can pass a size-1 sentinel buffer when direction gradients are not
// required (ctx.needs_input_grad[0]==False).

#include <metal_stdlib>

using namespace metal;

kernel void gsplat_spherical_harmonics_bwd(
    device const float* dirs           [[buffer(0)]],  // [N, 3]
    device const float* coeffs         [[buffer(1)]],  // [N, K, 3]
    device const uchar* masks          [[buffer(2)]],  // [N] or sentinel
    device const float* v_colors       [[buffer(3)]],  // [N, 3]
    device float*       v_coeffs_out   [[buffer(4)]],  // [N, K, 3]
    device float*       v_dirs_out     [[buffer(5)]],  // [N, 3] or sentinel
    constant uint&      N              [[buffer(6)]],
    constant uint&      K              [[buffer(7)]],
    constant uint&      degrees        [[buffer(8)]],
    constant uint&      has_mask       [[buffer(9)]],
    constant uint&      compute_v_dirs [[buffer(10)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= N) {
        return;
    }
    if (has_mask != 0u && masks[idx] == 0u) {
        // v_coeffs_out and v_dirs_out were pre-zeroed by the Python wrapper.
        return;
    }

    const uint coff = idx * K * 3u;
    const uint doff = idx * 3u;

    float3 vc = float3(v_colors[doff + 0u], v_colors[doff + 1u], v_colors[doff + 2u]);

    // Band 0: v_coeffs[0, :] = 0.2820947917738781 * v_colors.
    {
        float3 g = 0.2820947917738781f * vc;
        v_coeffs_out[coff + 0u] = g.x;
        v_coeffs_out[coff + 1u] = g.y;
        v_coeffs_out[coff + 2u] = g.z;
    }

    if (degrees < 1u) {
        // Caller pre-zeroed the remaining v_coeffs entries. Dir gradient
        // through band 0 is zero.
        return;
    }

    // Normalise dir and compute (x, y, z).
    float dx = dirs[doff + 0u];
    float dy = dirs[doff + 1u];
    float dz = dirs[doff + 2u];
    float inorm = rsqrt(dx * dx + dy * dy + dz * dz);
    float x = dx * inorm;
    float y = dy * inorm;
    float z = dz * inorm;

    float v_x = 0.0f, v_y = 0.0f, v_z = 0.0f;
    const bool want_vdir = (compute_v_dirs != 0u);

    // ---- Band 1 (coeff indices 1..3) ----
    float3 c_1 = float3(coeffs[coff + 1u*3u + 0u], coeffs[coff + 1u*3u + 1u], coeffs[coff + 1u*3u + 2u]);
    float3 c_2 = float3(coeffs[coff + 2u*3u + 0u], coeffs[coff + 2u*3u + 1u], coeffs[coff + 2u*3u + 2u]);
    float3 c_3 = float3(coeffs[coff + 3u*3u + 0u], coeffs[coff + 3u*3u + 1u], coeffs[coff + 3u*3u + 2u]);
    {
        const float c1 = 0.48860251190292f;
        float3 g1 = -c1 * y * vc;
        float3 g2 =  c1 * z * vc;
        float3 g3 = -c1 * x * vc;
        v_coeffs_out[coff + 1u*3u + 0u] = g1.x; v_coeffs_out[coff + 1u*3u + 1u] = g1.y; v_coeffs_out[coff + 1u*3u + 2u] = g1.z;
        v_coeffs_out[coff + 2u*3u + 0u] = g2.x; v_coeffs_out[coff + 2u*3u + 1u] = g2.y; v_coeffs_out[coff + 2u*3u + 2u] = g2.z;
        v_coeffs_out[coff + 3u*3u + 0u] = g3.x; v_coeffs_out[coff + 3u*3u + 1u] = g3.y; v_coeffs_out[coff + 3u*3u + 2u] = g3.z;
        if (want_vdir) {
            float cvc1 = dot(vc, c_1);
            float cvc2 = dot(vc, c_2);
            float cvc3 = dot(vc, c_3);
            v_x += -c1 * cvc3;
            v_y += -c1 * cvc1;
            v_z +=  c1 * cvc2;
        }
    }

    if (degrees < 2u) {
        if (want_vdir) {
            float3 dir_n = float3(x, y, z);
            float3 v_dir_n = float3(v_x, v_y, v_z);
            float3 v_d = (v_dir_n - dot(v_dir_n, dir_n) * dir_n) * inorm;
            v_dirs_out[doff + 0u] = v_d.x;
            v_dirs_out[doff + 1u] = v_d.y;
            v_dirs_out[doff + 2u] = v_d.z;
        }
        return;
    }

    // ---- Band 2 (coeff indices 4..8) ----
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
    {
        float3 g4 = pSH4 * vc;
        float3 g5 = pSH5 * vc;
        float3 g6 = pSH6 * vc;
        float3 g7 = pSH7 * vc;
        float3 g8 = pSH8 * vc;
        v_coeffs_out[coff + 4u*3u + 0u] = g4.x; v_coeffs_out[coff + 4u*3u + 1u] = g4.y; v_coeffs_out[coff + 4u*3u + 2u] = g4.z;
        v_coeffs_out[coff + 5u*3u + 0u] = g5.x; v_coeffs_out[coff + 5u*3u + 1u] = g5.y; v_coeffs_out[coff + 5u*3u + 2u] = g5.z;
        v_coeffs_out[coff + 6u*3u + 0u] = g6.x; v_coeffs_out[coff + 6u*3u + 1u] = g6.y; v_coeffs_out[coff + 6u*3u + 2u] = g6.z;
        v_coeffs_out[coff + 7u*3u + 0u] = g7.x; v_coeffs_out[coff + 7u*3u + 1u] = g7.y; v_coeffs_out[coff + 7u*3u + 2u] = g7.z;
        v_coeffs_out[coff + 8u*3u + 0u] = g8.x; v_coeffs_out[coff + 8u*3u + 1u] = g8.y; v_coeffs_out[coff + 8u*3u + 2u] = g8.z;
    }

    // Derivatives of band 2 bases used by both band-2 and band-3 dir-grads.
    float fTmp0B_z = -1.092548430592079f;
    float fC1_x = 2.0f * x;
    float fC1_y = -2.0f * y;
    float fS1_x = 2.0f * y;
    float fS1_y = 2.0f * x;
    float pSH6_z = 2.0f * 0.9461746957575601f * z;
    float pSH7_x = fTmp0B;
    float pSH7_z = fTmp0B_z * x;
    float pSH5_y = fTmp0B;
    float pSH5_z = fTmp0B_z * y;
    float pSH8_x = 0.5462742152960395f * fC1_x;
    float pSH8_y = 0.5462742152960395f * fC1_y;
    float pSH4_x = 0.5462742152960395f * fS1_x;
    float pSH4_y = 0.5462742152960395f * fS1_y;

    if (want_vdir) {
        float cvc4 = dot(vc, c_4);
        float cvc5 = dot(vc, c_5);
        float cvc6 = dot(vc, c_6);
        float cvc7 = dot(vc, c_7);
        float cvc8 = dot(vc, c_8);
        v_x += pSH4_x * cvc4 + pSH8_x * cvc8 + pSH7_x * cvc7;
        v_y += pSH4_y * cvc4 + pSH8_y * cvc8 + pSH5_y * cvc5;
        v_z += pSH6_z * cvc6 + pSH7_z * cvc7 + pSH5_z * cvc5;
    }

    if (degrees < 3u) {
        if (want_vdir) {
            float3 dir_n = float3(x, y, z);
            float3 v_dir_n = float3(v_x, v_y, v_z);
            float3 v_d = (v_dir_n - dot(v_dir_n, dir_n) * dir_n) * inorm;
            v_dirs_out[doff + 0u] = v_d.x;
            v_dirs_out[doff + 1u] = v_d.y;
            v_dirs_out[doff + 2u] = v_d.z;
        }
        return;
    }

    // ---- Band 3 (coeff indices 9..15) ----
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
    {
        float3 g9  = pSH9  * vc;
        float3 g10 = pSH10 * vc;
        float3 g11 = pSH11 * vc;
        float3 g12 = pSH12 * vc;
        float3 g13 = pSH13 * vc;
        float3 g14 = pSH14 * vc;
        float3 g15 = pSH15 * vc;
        v_coeffs_out[coff +  9u*3u + 0u] = g9.x;  v_coeffs_out[coff +  9u*3u + 1u] = g9.y;  v_coeffs_out[coff +  9u*3u + 2u] = g9.z;
        v_coeffs_out[coff + 10u*3u + 0u] = g10.x; v_coeffs_out[coff + 10u*3u + 1u] = g10.y; v_coeffs_out[coff + 10u*3u + 2u] = g10.z;
        v_coeffs_out[coff + 11u*3u + 0u] = g11.x; v_coeffs_out[coff + 11u*3u + 1u] = g11.y; v_coeffs_out[coff + 11u*3u + 2u] = g11.z;
        v_coeffs_out[coff + 12u*3u + 0u] = g12.x; v_coeffs_out[coff + 12u*3u + 1u] = g12.y; v_coeffs_out[coff + 12u*3u + 2u] = g12.z;
        v_coeffs_out[coff + 13u*3u + 0u] = g13.x; v_coeffs_out[coff + 13u*3u + 1u] = g13.y; v_coeffs_out[coff + 13u*3u + 2u] = g13.z;
        v_coeffs_out[coff + 14u*3u + 0u] = g14.x; v_coeffs_out[coff + 14u*3u + 1u] = g14.y; v_coeffs_out[coff + 14u*3u + 2u] = g14.z;
        v_coeffs_out[coff + 15u*3u + 0u] = g15.x; v_coeffs_out[coff + 15u*3u + 1u] = g15.y; v_coeffs_out[coff + 15u*3u + 2u] = g15.z;
    }

    float fTmp0C_z = -2.285228997322329f * 2.0f * z;
    float fTmp1B_z = 1.445305721320277f;
    float fC2_x = fC1 + x * fC1_x - y * fS1_x;
    float fC2_y = x * fC1_y - fS1 - y * fS1_y;
    float fS2_x = fS1 + x * fS1_x + y * fC1_x;
    float fS2_y = x * fS1_y + fC1 + y * fC1_y;
    float pSH12_z = 3.0f * 1.865881662950577f * z2 - 1.119528997770346f;
    float pSH13_x = fTmp0C;
    float pSH13_z = fTmp0C_z * x;
    float pSH11_y = fTmp0C;
    float pSH11_z = fTmp0C_z * y;
    float pSH14_x = fTmp1B * fC1_x;
    float pSH14_y = fTmp1B * fC1_y;
    float pSH14_z = fTmp1B_z * fC1;
    float pSH10_x = fTmp1B * fS1_x;
    float pSH10_y = fTmp1B * fS1_y;
    float pSH10_z = fTmp1B_z * fS1;
    float pSH15_x = -0.5900435899266435f * fC2_x;
    float pSH15_y = -0.5900435899266435f * fC2_y;
    float pSH9_x  = -0.5900435899266435f * fS2_x;
    float pSH9_y  = -0.5900435899266435f * fS2_y;

    if (want_vdir) {
        float cvc9  = dot(vc, c_9);
        float cvc10 = dot(vc, c_10);
        float cvc11 = dot(vc, c_11);
        float cvc12 = dot(vc, c_12);
        float cvc13 = dot(vc, c_13);
        float cvc14 = dot(vc, c_14);
        float cvc15 = dot(vc, c_15);
        v_x += pSH9_x  * cvc9  + pSH15_x * cvc15 + pSH10_x * cvc10 + pSH14_x * cvc14 + pSH13_x * cvc13;
        v_y += pSH9_y  * cvc9  + pSH15_y * cvc15 + pSH10_y * cvc10 + pSH14_y * cvc14 + pSH11_y * cvc11;
        v_z += pSH12_z * cvc12 + pSH13_z * cvc13 + pSH11_z * cvc11 + pSH14_z * cvc14 + pSH10_z * cvc10;
    }

    if (degrees < 4u) {
        if (want_vdir) {
            float3 dir_n = float3(x, y, z);
            float3 v_dir_n = float3(v_x, v_y, v_z);
            float3 v_d = (v_dir_n - dot(v_dir_n, dir_n) * dir_n) * inorm;
            v_dirs_out[doff + 0u] = v_d.x;
            v_dirs_out[doff + 1u] = v_d.y;
            v_dirs_out[doff + 2u] = v_d.z;
        }
        return;
    }

    // ---- Band 4 (coeff indices 16..24) ----
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
    {
        float3 g16 = pSH16 * vc;
        float3 g17 = pSH17 * vc;
        float3 g18 = pSH18 * vc;
        float3 g19 = pSH19 * vc;
        float3 g20 = pSH20 * vc;
        float3 g21 = pSH21 * vc;
        float3 g22 = pSH22 * vc;
        float3 g23 = pSH23 * vc;
        float3 g24 = pSH24 * vc;
        v_coeffs_out[coff + 16u*3u + 0u] = g16.x; v_coeffs_out[coff + 16u*3u + 1u] = g16.y; v_coeffs_out[coff + 16u*3u + 2u] = g16.z;
        v_coeffs_out[coff + 17u*3u + 0u] = g17.x; v_coeffs_out[coff + 17u*3u + 1u] = g17.y; v_coeffs_out[coff + 17u*3u + 2u] = g17.z;
        v_coeffs_out[coff + 18u*3u + 0u] = g18.x; v_coeffs_out[coff + 18u*3u + 1u] = g18.y; v_coeffs_out[coff + 18u*3u + 2u] = g18.z;
        v_coeffs_out[coff + 19u*3u + 0u] = g19.x; v_coeffs_out[coff + 19u*3u + 1u] = g19.y; v_coeffs_out[coff + 19u*3u + 2u] = g19.z;
        v_coeffs_out[coff + 20u*3u + 0u] = g20.x; v_coeffs_out[coff + 20u*3u + 1u] = g20.y; v_coeffs_out[coff + 20u*3u + 2u] = g20.z;
        v_coeffs_out[coff + 21u*3u + 0u] = g21.x; v_coeffs_out[coff + 21u*3u + 1u] = g21.y; v_coeffs_out[coff + 21u*3u + 2u] = g21.z;
        v_coeffs_out[coff + 22u*3u + 0u] = g22.x; v_coeffs_out[coff + 22u*3u + 1u] = g22.y; v_coeffs_out[coff + 22u*3u + 2u] = g22.z;
        v_coeffs_out[coff + 23u*3u + 0u] = g23.x; v_coeffs_out[coff + 23u*3u + 1u] = g23.y; v_coeffs_out[coff + 23u*3u + 2u] = g23.z;
        v_coeffs_out[coff + 24u*3u + 0u] = g24.x; v_coeffs_out[coff + 24u*3u + 1u] = g24.y; v_coeffs_out[coff + 24u*3u + 2u] = g24.z;
    }

    if (want_vdir) {
        float fTmp0D_z = 3.0f * -4.683325804901025f * z2 + 2.007139630671868f;
        float fTmp1C_z = 2.0f * 3.31161143515146f * z;
        float fTmp2B_z = -1.770130769779931f;
        float fC3_x = fC2 + x * fC2_x - y * fS2_x;
        float fC3_y = x * fC2_y - fS2 - y * fS2_y;
        float fS3_x = fS2 + y * fC2_x + x * fS2_x;
        float fS3_y = x * fS2_y + fC2 + y * fC2_y;
        float pSH20_z = 1.984313483298443f * (pSH12 + z * pSH12_z) - 1.006230589874905f * pSH6_z;
        float pSH21_x = fTmp0D;
        float pSH21_z = fTmp0D_z * x;
        float pSH19_y = fTmp0D;
        float pSH19_z = fTmp0D_z * y;
        float pSH22_x = fTmp1C * fC1_x;
        float pSH22_y = fTmp1C * fC1_y;
        float pSH22_z = fTmp1C_z * fC1;
        float pSH18_x = fTmp1C * fS1_x;
        float pSH18_y = fTmp1C * fS1_y;
        float pSH18_z = fTmp1C_z * fS1;
        float pSH23_x = fTmp2B * fC2_x;
        float pSH23_y = fTmp2B * fC2_y;
        float pSH23_z = fTmp2B_z * fC2;
        float pSH17_x = fTmp2B * fS2_x;
        float pSH17_y = fTmp2B * fS2_y;
        float pSH17_z = fTmp2B_z * fS2;
        float pSH24_x = 0.6258357354491763f * fC3_x;
        float pSH24_y = 0.6258357354491763f * fC3_y;
        float pSH16_x = 0.6258357354491763f * fS3_x;
        float pSH16_y = 0.6258357354491763f * fS3_y;

        float cvc16 = dot(vc, c_16);
        float cvc17 = dot(vc, c_17);
        float cvc18 = dot(vc, c_18);
        float cvc19 = dot(vc, c_19);
        float cvc20 = dot(vc, c_20);
        float cvc21 = dot(vc, c_21);
        float cvc22 = dot(vc, c_22);
        float cvc23 = dot(vc, c_23);
        float cvc24 = dot(vc, c_24);

        v_x += pSH16_x * cvc16 + pSH24_x * cvc24 + pSH17_x * cvc17 + pSH23_x * cvc23
             + pSH18_x * cvc18 + pSH22_x * cvc22 + pSH21_x * cvc21;
        v_y += pSH16_y * cvc16 + pSH24_y * cvc24 + pSH17_y * cvc17 + pSH23_y * cvc23
             + pSH18_y * cvc18 + pSH22_y * cvc22 + pSH19_y * cvc19;
        v_z += pSH20_z * cvc20 + pSH21_z * cvc21 + pSH19_z * cvc19 + pSH22_z * cvc22
             + pSH18_z * cvc18 + pSH23_z * cvc23 + pSH17_z * cvc17;

        float3 dir_n = float3(x, y, z);
        float3 v_dir_n = float3(v_x, v_y, v_z);
        float3 v_d = (v_dir_n - dot(v_dir_n, dir_n) * dir_n) * inorm;
        v_dirs_out[doff + 0u] = v_d.x;
        v_dirs_out[doff + 1u] = v_d.y;
        v_dirs_out[doff + 2u] = v_d.z;
    }
}
