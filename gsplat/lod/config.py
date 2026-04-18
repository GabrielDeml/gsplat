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

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class LoDConfig:
    """Configuration for the LoD (Level-of-Detail) training pipeline."""

    # --- Stage 1 (coarse pre-training) ---------------------------------------
    coarse_iters: int = 100_000
    coarse_cap_max: int = 5_000_000
    coarse_sh_degree_schedule: List[Tuple[int, int]] = field(
        default_factory=lambda: [(0, 0), (20_000, 1), (40_000, 3)]
    )

    # --- Hierarchy / HSPT ----------------------------------------------------
    hspt_size_threshold: float = 1e-3  # volume s1*s2*s3 for upper/lower split

    # --- LoD selection -------------------------------------------------------
    # We use m_d(i) = T * max_j(s_i^j) for Eq. 3 or
    #         m_d(i) = T * sqrt(s1^2 s2^2 + s1^2 s3^2 + s2^2 s3^2) for Eq. 6.
    # Cut condition: include i (and stop) iff ||mu_i - p_cam|| >= m_d(i); else
    # descend. Under merge math child m_d < parent m_d (heap), so BFS is proper.
    T: float = 1e-2
    use_refined_m_d: bool = False  # False -> Eq. 3 simple, True -> Eq. 6
    frustum_radius_mult: float = 3.0  # sphere radius = mult * max(scale_activated)

    # --- GPU cache -----------------------------------------------------------
    cache_capacity_gaussians: int = 15_000_000
    cache_D_min: float = 0.8
    cache_D_max: float = 1.25
    cache_full_flush_every: int = 1000

    # --- View selection ------------------------------------------------------
    knn_k: int = 32
    knn_W: float = 1.0
    uniform_view_every: int = 128

    # --- Stage 2 (LoD training) ---------------------------------------------
    lod_iters: int = 150_000
    lod_cap_max: int = 150_000_000
    densify_every: int = 300
    densify_start_iter: int = 1_000
    densify_stop_iter: int = 120_000
    spawn_grad_threshold: float = 2e-4
    respawn_opacity_threshold: float = 0.005
    respawn_unused_iters: int = 600
    lr_noise_factor: float = 5.0  # paper A.5 distance noise factor on cache lookup

    # --- Skybox --------------------------------------------------------------
    skybox_enabled: bool = True
    skybox_n_points: int = 100_000
    skybox_radius: float = 100.0

    # --- SH degree during streaming ----------------------------------------
    sh_degree: int = 3
    sh_degree_schedule: List[Tuple[int, int]] = field(
        default_factory=lambda: [(0, 0), (15_000, 1), (30_000, 3)]
    )

    # --- Device selection --------------------------------------------------
    device: str = "cuda"
