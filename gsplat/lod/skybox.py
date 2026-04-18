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

"""Persistent skybox Gaussian set.

These Gaussians sit outside the hierarchy and are always rendered. They cover
the distant background (sky dome) and give every training view some content
beyond the explicit LoD set.
"""

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor


@dataclass
class SkyboxSet:
    """A fixed set of sky Gaussians kept GPU-resident throughout training."""

    means: Tensor
    scales: Tensor
    quats: Tensor
    opacities: Tensor
    sh0: Tensor
    shN: Tensor

    @classmethod
    def make_icosphere(
        cls,
        n_points: int,
        radius: float,
        centre: Tensor,
        sh_degree: int = 3,
        scale_init: float = 0.1,
        device: str | torch.device = "cuda",
    ) -> "SkyboxSet":
        """Fibonacci-sphere sampling of the skybox.

        Points are placed on the sphere of radius ``radius`` centred at
        ``centre``, with isotropic log-scales of ``log(scale_init * radius)``.
        """
        # Fibonacci lattice
        idx = torch.arange(n_points, dtype=torch.float32)
        phi = torch.acos(1.0 - 2.0 * (idx + 0.5) / n_points)
        golden = torch.pi * (3.0 - 5.0 ** 0.5)
        theta = golden * idx
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        dirs = torch.stack([x, y, z], dim=-1)  # [N, 3]
        means = centre.to(device).view(1, 3) + radius * dirs.to(device)

        scale_log = torch.log(torch.tensor(scale_init * radius, device=device))
        scales = scale_log * torch.ones(n_points, 3, device=device)
        quats = torch.zeros(n_points, 4, device=device)
        quats[:, 0] = 1.0
        opacities = torch.full((n_points,), 2.0, device=device)

        K = (sh_degree + 1) ** 2 - 1
        sh0 = torch.zeros(n_points, 1, 3, device=device)
        shN = torch.zeros(n_points, K, 3, device=device)

        return cls(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh0=sh0,
            shN=shN,
        )

    def as_dict(self) -> Dict[str, Tensor]:
        return {
            "means": self.means,
            "scales": self.scales,
            "quats": self.quats,
            "opacities": self.opacities,
            "sh0": self.sh0,
            "shN": self.shN,
        }
