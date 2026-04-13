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

"""Shared types, enums, and pure-Python parameter classes used by both the
CUDA and MPS backends.

These are intentionally free of any backend dependency so that the pure-PyTorch
reference implementations (``_torch_impl*.py``) can import them without
triggering CUDA or Metal extension loading.
"""

from abc import ABC
from enum import IntEnum
from typing import List

from typing_extensions import Literal

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ExternalDistortionModelMeta = Literal["bivariate-windshield"]
CameraModel = Literal["pinhole", "ortho", "fisheye", "ftheta", "lidar"]

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RollingShutterType(IntEnum):
    ROLLING_TOP_TO_BOTTOM = 0
    ROLLING_LEFT_TO_RIGHT = 1
    ROLLING_BOTTOM_TO_TOP = 2
    ROLLING_RIGHT_TO_LEFT = 3
    GLOBAL = 4


class FThetaPolynomialType(IntEnum):
    PIXELDIST_TO_ANGLE = 0
    ANGLE_TO_PIXELDIST = 1


class ExternalDistortionReferencePolynomial(IntEnum):
    FORWARD = 1
    BACKWARD = 2


# ---------------------------------------------------------------------------
# ABC base
# ---------------------------------------------------------------------------


class ExternalDistortionModelParameters(ABC):
    """Base class for external distortion model parameters.

    All concrete external distortion models (e.g. BivariateWindshieldModelParameters)
    should inherit from this class so that the rendering API can accept any
    distortion model through a single type-erased parameter.
    """


# ---------------------------------------------------------------------------
# Pure-Python parameter classes
#
# On the CUDA backend these are typically backed by C++ custom classes
# (``torch::CustomClassHolder``).  The pure-Python versions here carry the
# same attributes and default values so that the reference implementations
# and the MPS backend can use them without a native extension.
# ---------------------------------------------------------------------------


class UnscentedTransformParameters:
    """Pure-Python equivalent of the C++ ``UnscentedTransformParameters``.

    Default values match the C++ struct in ``gsplat/cuda/include/Cameras.h``.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 2.0,
        kappa: float = 0.0,
        in_image_margin_factor: float = 0.1,
        require_all_sigma_points_valid: bool = False,
    ):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.in_image_margin_factor = in_image_margin_factor
        self.require_all_sigma_points_valid = require_all_sigma_points_valid


class FThetaCameraDistortionParameters:
    """Pure-Python equivalent of the C++ ``FThetaCameraDistortionParameters``."""

    PolynomialDegree = 15

    def __init__(
        self,
        reference_poly: int = 0,
        pixeldist_to_angle_poly: List[float] = None,
        angle_to_pixeldist_poly: List[float] = None,
        max_angle: float = 0.0,
        linear_cde: List[float] = None,
    ):
        self.reference_poly = reference_poly
        self.pixeldist_to_angle_poly = (
            pixeldist_to_angle_poly
            if pixeldist_to_angle_poly is not None
            else [0.0] * self.PolynomialDegree
        )
        self.angle_to_pixeldist_poly = (
            angle_to_pixeldist_poly
            if angle_to_pixeldist_poly is not None
            else [0.0] * self.PolynomialDegree
        )
        self.max_angle = max_angle
        self.linear_cde = linear_cde if linear_cde is not None else [0.0, 0.0, 0.0]


class BivariateWindshieldModelParameters(ExternalDistortionModelParameters):
    """Pure-Python equivalent of the C++ ``BivariateWindshieldModelParameters``.

    Constants match the C++ defaults.
    """

    MAX_ORDER: int = 5
    MAX_COEFFS: int = 21
