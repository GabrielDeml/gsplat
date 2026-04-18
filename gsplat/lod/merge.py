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

"""Parent-from-two-children merge math for the Gaussian hierarchy.

For a merge of children A, B into parent P:
    w_i     = sigmoid(opacity_i) * volume_i            (visibility * mass weight)
    mu_P    = (w_A mu_A + w_B mu_B) / (w_A + w_B)
    Sigma_P = sum_i w_i (Sigma_i + d_i d_i^T) / sum_i w_i,  d_i = mu_i - mu_P
    op_P    = logit(sigma_A + sigma_B - sigma_A sigma_B)      (probabilistic union)
    sh_P    = (w_A sh_A + w_B sh_B) / (w_A + w_B)

We then eigendecompose Sigma_P = R diag(eig) R^T and read back:
    scale_P = log(sqrt(eig))
    quat_P  = from_rotation_matrix(R)

Inputs are stored in gsplat's pre-activation convention (log-scales, logit-
opacities, raw quats). Outputs match.
"""

from typing import Dict, Tuple

import torch
from torch import Tensor

from ..utils import normalized_quat_to_rotmat


_EPS = 1e-8
_OPACITY_CLAMP = (-20.0, 20.0)


def _safe_logit(p: Tensor) -> Tensor:
    p = p.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(p / (1.0 - p))


def _quat_from_rotmat(R: Tensor) -> Tensor:
    """Convert a batch of rotation matrices [..., 3, 3] to quaternions [..., 4]
    in wxyz order, matching gsplat's convention.

    Uses the standard Shepperd method.
    """
    assert R.shape[-2:] == (3, 3)
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = m00 + m11 + m22

    t_w = 1.0 + trace
    t_x = 1.0 + m00 - m11 - m22
    t_y = 1.0 - m00 + m11 - m22
    t_z = 1.0 - m00 - m11 + m22

    # Pick the branch with largest squared component for numerical stability.
    stacked = torch.stack([t_w, t_x, t_y, t_z], dim=-1)
    branch = stacked.argmax(dim=-1)

    # Compute quat per branch, then gather.
    out = torch.empty(R.shape[:-2] + (4,), dtype=R.dtype, device=R.device)

    # Branch W
    s_w = torch.sqrt(t_w.clamp_min(_EPS)) * 2.0
    qw_w = 0.25 * s_w
    qx_w = (m21 - m12) / s_w
    qy_w = (m02 - m20) / s_w
    qz_w = (m10 - m01) / s_w

    # Branch X
    s_x = torch.sqrt(t_x.clamp_min(_EPS)) * 2.0
    qw_x = (m21 - m12) / s_x
    qx_x = 0.25 * s_x
    qy_x = (m01 + m10) / s_x
    qz_x = (m02 + m20) / s_x

    # Branch Y
    s_y = torch.sqrt(t_y.clamp_min(_EPS)) * 2.0
    qw_y = (m02 - m20) / s_y
    qx_y = (m01 + m10) / s_y
    qy_y = 0.25 * s_y
    qz_y = (m12 + m21) / s_y

    # Branch Z
    s_z = torch.sqrt(t_z.clamp_min(_EPS)) * 2.0
    qw_z = (m10 - m01) / s_z
    qx_z = (m02 + m20) / s_z
    qy_z = (m12 + m21) / s_z
    qz_z = 0.25 * s_z

    qw = torch.where(branch == 0, qw_w, torch.where(branch == 1, qw_x, torch.where(branch == 2, qw_y, qw_z)))
    qx = torch.where(branch == 0, qx_w, torch.where(branch == 1, qx_x, torch.where(branch == 2, qx_y, qx_z)))
    qy = torch.where(branch == 0, qy_w, torch.where(branch == 1, qy_x, torch.where(branch == 2, qy_y, qy_z)))
    qz = torch.where(branch == 0, qz_w, torch.where(branch == 1, qz_x, torch.where(branch == 2, qz_y, qz_z)))

    out[..., 0] = qw
    out[..., 1] = qx
    out[..., 2] = qy
    out[..., 3] = qz

    # Normalise (should already be unit-norm modulo float error).
    out = out / out.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    # Canonicalise sign (make w >= 0) for determinism.
    flip = (out[..., 0] < 0).unsqueeze(-1).to(out.dtype)
    out = out * (1.0 - 2.0 * flip)
    return out


def _params_to_covariance(scale_log: Tensor, quat_raw: Tensor) -> Tensor:
    """Compute 3x3 covariance(s) from log-scale + raw quat. Batched."""
    s = torch.exp(scale_log)  # [..., 3]
    q = quat_raw / quat_raw.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    R = normalized_quat_to_rotmat(q)  # [..., 3, 3]
    S2 = (s * s).unsqueeze(-2) * torch.eye(3, dtype=s.dtype, device=s.device)  # [..., 3, 3]
    return R @ S2 @ R.transpose(-1, -2)


def _covariance_to_params(sigma: Tensor) -> Tuple[Tensor, Tensor]:
    """Eigen-decompose covariance(s) and return (log_scale, quat_raw).

    Ensures the rotation matrix has det(+1) (right-handed) by flipping an axis
    if necessary.
    """
    # symeig for symmetric tensor
    eigvals, eigvecs = torch.linalg.eigh(sigma)
    # eigvals ascending; scales are sqrt(eigvals).
    eigvals = eigvals.clamp_min(_EPS)
    # Ensure right-handed: det(eigvecs) should be +1; if -1 flip last column.
    det = torch.linalg.det(eigvecs)
    flip = (det < 0).to(eigvecs.dtype) * -1.0 + (det >= 0).to(eigvecs.dtype) * 1.0
    eigvecs = eigvecs.clone()
    eigvecs[..., :, -1] = eigvecs[..., :, -1] * flip.unsqueeze(-1)

    scales = torch.sqrt(eigvals)
    log_scales = torch.log(scales.clamp_min(_EPS))
    quats = _quat_from_rotmat(eigvecs)
    return log_scales, quats


@torch.no_grad()
def merge_pair_batch(a: Dict[str, Tensor], b: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Batched merge: each input dict holds params for ``K`` Gaussians; returns
    params for the ``K`` merged parents.

    Required keys: means, scales, quats, opacities, sh0, shN.
    All tensors on the same device (CPU is fine).
    """
    mu_a, mu_b = a["means"], b["means"]  # [K, 3]
    sc_a, sc_b = a["scales"], b["scales"]  # [K, 3] log
    q_a, q_b = a["quats"], b["quats"]  # [K, 4]
    op_a, op_b = a["opacities"], b["opacities"]  # [K]
    sh0_a, sh0_b = a["sh0"], b["sh0"]  # [K, 1, 3]
    shN_a, shN_b = a["shN"], b["shN"]  # [K, K_sh, 3]

    sigma_a = torch.sigmoid(op_a).clamp(1e-6, 1.0 - 1e-6)  # [K]
    sigma_b = torch.sigmoid(op_b).clamp(1e-6, 1.0 - 1e-6)

    vol_a = torch.exp(sc_a).prod(dim=-1)  # [K]
    vol_b = torch.exp(sc_b).prod(dim=-1)

    # Weights: visibility * mass. If both are near-zero, fall back to uniform.
    w_a = sigma_a * vol_a
    w_b = sigma_b * vol_b
    w_sum = w_a + w_b
    degen = w_sum < _EPS
    w_a = torch.where(degen, torch.full_like(w_a, 0.5), w_a / w_sum.clamp_min(_EPS))
    w_b = 1.0 - w_a

    wa = w_a.unsqueeze(-1)
    wb = w_b.unsqueeze(-1)
    mu_p = wa * mu_a + wb * mu_b

    # Covariances
    Sigma_a = _params_to_covariance(sc_a, q_a)  # [K, 3, 3]
    Sigma_b = _params_to_covariance(sc_b, q_b)
    d_a = (mu_a - mu_p).unsqueeze(-1)  # [K, 3, 1]
    d_b = (mu_b - mu_p).unsqueeze(-1)
    outer_a = d_a @ d_a.transpose(-1, -2)
    outer_b = d_b @ d_b.transpose(-1, -2)
    wa3 = w_a.view(-1, 1, 1)
    wb3 = w_b.view(-1, 1, 1)
    Sigma_p = wa3 * (Sigma_a + outer_a) + wb3 * (Sigma_b + outer_b)

    # Symmetrise for numerical hygiene.
    Sigma_p = 0.5 * (Sigma_p + Sigma_p.transpose(-1, -2))

    log_scale_p, quat_p = _covariance_to_params(Sigma_p)

    # Opacity: probabilistic union on activated values; store logit.
    sigma_p = (sigma_a + sigma_b - sigma_a * sigma_b).clamp(1e-6, 1.0 - 1e-6)
    # Clamp logit to avoid inf when near 1 after union.
    op_p = _safe_logit(sigma_p).clamp(*_OPACITY_CLAMP)

    wa1 = w_a.view(-1, 1, 1)
    wb1 = w_b.view(-1, 1, 1)
    sh0_p = wa1 * sh0_a + wb1 * sh0_b
    if shN_a.shape[1] > 0:
        shN_p = wa1 * shN_a + wb1 * shN_b
    else:
        shN_p = shN_a.new_zeros(shN_a.shape)

    return {
        "means": mu_p,
        "scales": log_scale_p,
        "quats": quat_p,
        "opacities": op_p,
        "sh0": sh0_p,
        "shN": shN_p,
    }


def _to_batch_of_one(d: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Promote per-Gaussian param dict to a [1, ...] batch. Expected per-key
    shapes (before): means [3], scales [3], quats [4], opacities scalar,
    sh0 [1, 3], shN [K_sh, 3]."""
    out: Dict[str, Tensor] = {}
    for k, v in d.items():
        if k == "opacities":
            out[k] = v.reshape(1)
        else:
            out[k] = v.unsqueeze(0)
    return out


@torch.no_grad()
def merge_pair(a: Dict[str, Tensor], b: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Merge a single pair of Gaussians. Thin wrapper over ``merge_pair_batch``."""
    out = merge_pair_batch(_to_batch_of_one(a), _to_batch_of_one(b))
    # Squeeze the batch dim, preserving opacities as a scalar tensor.
    result: Dict[str, Tensor] = {}
    for k, v in out.items():
        if k == "opacities":
            result[k] = v.squeeze(0)
        else:
            result[k] = v.squeeze(0)
    return result
