# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""
Stub build system for the MPS backend.

Currently a no-op — all operations use pure PyTorch implementations.
When native Metal shaders are added, this module will handle compilation
of .metal files and loading of Metal libraries.
"""


def build_and_load_gsplat():
    """Placeholder for future Metal shader compilation.

    Returns None since no native extension exists yet.

    .. todo:: MPS: Implement Metal shader compilation.
        This should:
        1. Discover ``.metal`` files in a ``csrc/`` subdirectory
        2. Compile them into a Metal library (``.metallib``)
        3. Load the library via PyTorch's MPS extension mechanism
        4. Register operators under ``torch.ops.gsplat`` (matching the CUDA API)

        Reference: ``gsplat/cuda/build.py`` for the CUDA JIT compilation flow.
    """
    return None
