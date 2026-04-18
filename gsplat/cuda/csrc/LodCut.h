/*
 * SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace at {
class Tensor;
}

namespace gsplat {

// Kernel launch: per-sphere frustum test.
//   mu     [M, 3]
//   radius [M]
//   planes [6, 4]  (nx, ny, nz, d; normalised)
//   out    [M] bool (true iff sphere not fully outside any plane)
void launch_lod_sphere_in_frustum_kernel(
    const at::Tensor &mu,
    const at::Tensor &radius,
    const at::Tensor &planes,
    at::Tensor &out
);

// Kernel launch: per-SPT binary search.
//   entries_size_parent [E] sorted descending within each SPT (CSR)
//   offsets             [S + 1] CSR offsets into entries
//   touched_spt_ids     [K] long, which SPT each threshold applies to
//   distances           [K] float, camera distances for those SPTs
//   T                   scalar LoD threshold
//   Output:
//   cut_count [K] long, how many entries (prefix) each touched SPT keeps
//
// This is an easier kernel than level-synchronous BFS and already beats the
// Python loop by a big margin when K is large.
void launch_lod_spt_cut_count_kernel(
    const at::Tensor &entries_size_parent,
    const at::Tensor &offsets,
    const at::Tensor &touched_spt_ids,
    const at::Tensor &distances,
    double T,
    at::Tensor &cut_count
);

} // namespace gsplat
