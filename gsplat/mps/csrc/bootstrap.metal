#include <metal_stdlib>

using namespace metal;

kernel void gsplat_bootstrap_fill_float(
    device float* out [[buffer(0)]],
    constant float& value [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    out[index] = value;
}
