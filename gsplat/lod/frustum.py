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

"""Frustum-sphere intersection test for LoD BFS culling.

Given a pinhole camera (viewmat, K, width, height, near, far), extract the 6
view-frustum planes in world space and test whether a sphere (centre mu,
radius r) lies outside the frustum.

We use "outside if outside any plane" — conservative (false positives -> keep
the sphere) so we never cull something that should be drawn.
"""

from typing import Tuple

import torch
from torch import Tensor


def extract_frustum_planes(
    viewmat: Tensor, K: Tensor, width: int, height: int, near: float = 0.01, far: float = 1e6
) -> Tensor:
    """Return 6 frustum planes in world space as a [6, 4] tensor of (a,b,c,d)
    such that a point (x,y,z) is *inside* the half-space when a*x+b*y+c*z+d >= 0.

    Planes: near, far, left, right, top, bottom.

    Args:
        viewmat: [4, 4] world-to-camera transform.
        K: [3, 3] intrinsics.
        width, height: pixel dimensions.
        near, far: clip planes.
    """
    device = viewmat.device
    dtype = viewmat.dtype
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # Camera-space plane normals (pointing INTO the frustum).
    # Near plane:  z >= near  -> n = (0, 0, 1),  d = -near
    # Far plane:   z <= far   -> n = (0, 0, -1), d = far
    # Left:   fx * x / z >= -cx  -> x*fx + z*cx >= 0  -> n = (fx, 0, cx)
    # Right:  fx * x / z <=  (W - cx) -> x*fx - z*(W-cx) <= 0 -> n=(-fx,0,W-cx)
    # Bottom: fy * y / z >= -cy -> n = (0, fy, cy)
    # Top:    fy * y / z <= (H - cy) -> n = (0, -fy, H-cy)
    #
    # Each (n, d) describes cam-space half-space n . p_cam + d >= 0 with d=0
    # for the pinhole frustum planes (they pass through the origin) except
    # near/far.
    planes_cam = torch.tensor(
        [
            [0.0, 0.0, 1.0, -near],  # near
            [0.0, 0.0, -1.0, far],  # far
            [fx, 0.0, cx, 0.0],  # left
            [-fx, 0.0, width - cx, 0.0],  # right
            [0.0, fy, cy, 0.0],  # bottom
            [0.0, -fy, height - cy, 0.0],  # top
        ],
        dtype=dtype,
        device=device,
    )
    # Normalize each plane's normal to unit length so that plane.dot(point) is
    # the signed distance to the plane.
    n = planes_cam[:, :3]
    d = planes_cam[:, 3:]
    norm = n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    planes_cam = torch.cat([n / norm, d / norm], dim=-1)

    # Transform cam-space planes to world space. For a plane (n_cam, d_cam),
    # world-space plane is (R_wc @ n_cam, d_cam - (R_wc @ n_cam) . t_wc) where
    # viewmat = [R_cw | t_cw; 0 0 0 1] takes world -> cam, so cam -> world is
    # R_wc = R_cw^T and t_wc = -R_cw^T @ t_cw.
    R_cw = viewmat[:3, :3]
    t_cw = viewmat[:3, 3]
    R_wc = R_cw.transpose(-1, -2)
    t_wc = -(R_wc @ t_cw)

    n_cam = planes_cam[:, :3]  # [6, 3]
    d_cam = planes_cam[:, 3]  # [6]

    n_world = n_cam @ R_wc.transpose(-1, -2)  # [6, 3]; (R_wc @ n_cam.T).T
    d_world = d_cam - (n_world @ t_wc)  # [6]
    return torch.cat([n_world, d_world.unsqueeze(-1)], dim=-1)  # [6, 4]


def spheres_in_frustum(
    mu: Tensor,
    radius: Tensor,
    planes: Tensor,
) -> Tensor:
    """Conservative inside-frustum test for a batch of spheres.

    Args:
        mu:     [M, 3] sphere centres (world space).
        radius: [M]     sphere radii (world space).
        planes: [6, 4]  frustum planes from ``extract_frustum_planes``.

    Returns:
        [M] bool tensor, True when the sphere is NOT fully outside any plane
        (i.e., possibly visible).
    """
    # Signed distance from sphere centre to each plane = a*x + b*y + c*z + d.
    # Sphere is outside if dist < -radius for any plane.
    # Inside/touching: dist >= -radius for all planes.
    # planes: [6, 4]; mu: [M, 3]
    n = planes[:, :3]  # [6, 3]
    d = planes[:, 3]  # [6]
    # distances[m, p] = mu[m] . n[p] + d[p]
    distances = mu @ n.transpose(-1, -2) + d  # [M, 6]
    # Test: all distances >= -radius broadcast.
    return (distances >= -radius.unsqueeze(-1)).all(dim=-1)


def spheres_and_planes(
    mu: Tensor,
    radius: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    near: float = 0.01,
    far: float = 1e6,
) -> Tensor:
    """Convenience wrapper: extract planes and test spheres in one call."""
    planes = extract_frustum_planes(viewmat, K, width, height, near, far)
    return spheres_in_frustum(mu, radius, planes)
