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

"""Level-of-Detail Gaussian subsystem for gsplat.

Implements the out-of-core, hierarchical 3DGS pipeline from:
    A LoD of Gaussians: Out-of-Core Training and Rendering for Seamless
    Ultra-Large Scene Reconstruction.  Windisch et al., 2026.
    arXiv:2507.01110v4

See ``gsplat/lod/README.md`` for design notes (TODO) and the module docstrings
for each component. Sub-modules may be imported individually.
"""

from .config import LoDConfig
from .hierarchy import GaussianHierarchy
from .merge import merge_pair, merge_pair_batch
from .builder import build_hierarchy
from .frustum import extract_frustum_planes, spheres_in_frustum
from .hspt import HSPT, build_hspt
from .cut import compute_render_set, RenderSet
from .cut_gpu import compute_render_set_gpu
from .streaming import CpuGaussianStore
from .cache import GpuSptCache, CacheEntry
from .view_sampler import KnnViewSampler
from .skybox import SkyboxSet
from .optim import OutOfCoreAdam
from .densify import HierarchicalMcmcDensifier, GradStats

__all__ = [
    "LoDConfig",
    "GaussianHierarchy",
    "merge_pair",
    "merge_pair_batch",
    "build_hierarchy",
    "extract_frustum_planes",
    "spheres_in_frustum",
    "HSPT",
    "build_hspt",
    "compute_render_set",
    "compute_render_set_gpu",
    "RenderSet",
    "CpuGaussianStore",
    "GpuSptCache",
    "CacheEntry",
    "KnnViewSampler",
    "SkyboxSet",
    "OutOfCoreAdam",
    "HierarchicalMcmcDensifier",
    "GradStats",
]
