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

"""MPS backend loader for gsplat."""

import torch
from rich.console import Console

from .build import build_and_load_gsplat


def mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) backend is available."""

    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


_C = None

if mps_available():
    try:
        _C = build_and_load_gsplat()
    except Exception as exc:
        raise RuntimeError(
            "gsplat: native MPS shader setup failed. This machine reports MPS "
            "availability, so gsplat requires packaged Metal shader initialization "
            f"to succeed. {exc}"
        ) from exc
else:
    Console().print(
        "[yellow]gsplat: No MPS backend available. "
        "gsplat MPS will run on CPU via PyTorch fallbacks.[/yellow]"
    )

__all__ = ["_C", "mps_available"]
