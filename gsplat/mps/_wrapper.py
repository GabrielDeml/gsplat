# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""MPS backend wrapper for gsplat.

This module mirrors the public API of ``gsplat.cuda._wrapper`` but delegates
every operation to pure-PyTorch reference implementations so that the full
rasterization pipeline can run on Apple MPS (or any device without a CUDA
extension).
"""

import math
import warnings
from typing import Any, Callable, Optional, Tuple

import torch
from torch import Tensor
from gsplat._helper import assert_shape

# ---------------------------------------------------------------------------
# Shared types — imported from the backend-agnostic _types module.
# Re-exported here so that ``from gsplat.mps._wrapper import CameraModel``
# keeps working exactly like the CUDA wrapper.
# ---------------------------------------------------------------------------
from gsplat._types import (
    CameraModel,
    ExternalDistortionModelMeta,
    ExternalDistortionModelParameters,
    ExternalDistortionReferencePolynomial,
    RollingShutterType,
    FThetaPolynomialType,
    UnscentedTransformParameters,
    FThetaCameraDistortionParameters,
    BivariateWindshieldModelParameters,
)

# Shared device-agnostic modules from the cuda package (no CUDA loading triggered).
from gsplat.cuda._lidar import (
    SpinningDirection,
    LidarModelParameters,
    RowOffsetStructuredSpinningLidarModelParameters,
    RowOffsetStructuredSpinningLidarModelParametersExt as RowOffsetStructuredSpinningLidarModelParametersExtBase,
    FOV as FOVBase,
)

# Import the MPS backend eagerly on real MPS systems so ``import gsplat`` fails
# fast if packaged Metal shader setup is unavailable.
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    from ._backend import _C as _MPS_NATIVE_BACKEND  # noqa: F401

# ---------------------------------------------------------------------------
# Lidar parameter wrappers (no ``to_cpp()`` needed on MPS)
# ---------------------------------------------------------------------------


class FOV(FOVBase):
    @classmethod
    def from_base(cls, base: FOVBase) -> "FOV":
        return cls(start=base.start, span=base.span, direction=base.direction)


class RowOffsetStructuredSpinningLidarModelParametersExt(
    RowOffsetStructuredSpinningLidarModelParametersExtBase
):
    """Lidar camera parameters extended with acceleration structures (MPS).

    Unlike the CUDA variant this class does **not** expose a ``to_cpp()``
    method because no native extension is involved.
    """

    pass


# ---------------------------------------------------------------------------
# Feature detection
# ---------------------------------------------------------------------------


def has_3dgs() -> bool:
    return True


def has_2dgs() -> bool:
    return True


def has_3dgut() -> bool:
    return True


def has_adam() -> bool:
    return True


def has_camera_wrappers() -> bool:
    # TODO: MPS: Implement native camera wrappers as Metal kernels.
    #   CUDA equivalents: ``CameraWrappers.cu``, ``ExternalDistortionWrappers.cu``
    return False


def has_reloc() -> bool:
    # TODO: MPS: Implement relocation as a Metal kernel.
    #   CUDA equivalent: ``RelocationCUDA.cu``
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unavailable_mps_cls(name: str) -> Any:
    """Placeholder class for native CUDA-only wrappers on the MPS backend."""

    class _UnavailableMpsCls:
        __name__ = name

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "gsplat MPS backend does not expose a native class for "
                f"'{name}'. Use the pure-Python MPS equivalents instead."
            )

    return _UnavailableMpsCls


def _make_lazy_cuda_cls(name: str) -> Any:
    """Compatibility shim for tests/shared code that import this helper."""

    return globals().get(name, _unavailable_mps_cls(name))


def _get_mps_backend() -> Any:
    """Return the compiled MPS backend handle.

    The public MPS wrapper still uses pure-PyTorch implementations today, but
    future native Metal kernels will resolve through this helper.
    """

    # pylint: disable=import-outside-toplevel
    from ._backend import _C

    if _C is None:
        raise RuntimeError(
            "gsplat MPS backend is not initialized. Native MPS shaders are only "
            "available when torch.backends.mps.is_available() is true and gsplat "
            "successfully compiles its packaged Metal sources."
        )
    return _C


def _make_lazy_mps_shader(name: str) -> Callable:
    """Resolve a compiled Metal kernel lazily from the cached backend handle."""

    def call_mps_shader(*args: Any, **kwargs: Any) -> Any:
        return _get_mps_backend().get_kernel(name)(*args, **kwargs)

    return call_mps_shader


def _triu_to_full(triu: Tensor) -> Tensor:
    """Convert upper-triangle packed covariance ``[..., 6]`` to full ``[..., 3, 3]``."""
    mat = torch.zeros(
        *triu.shape[:-1], 3, 3, device=triu.device, dtype=triu.dtype
    )
    mat[..., 0, 0] = triu[..., 0]
    mat[..., 0, 1] = triu[..., 1]
    mat[..., 0, 2] = triu[..., 2]
    mat[..., 1, 0] = triu[..., 1]
    mat[..., 1, 1] = triu[..., 3]
    mat[..., 1, 2] = triu[..., 4]
    mat[..., 2, 0] = triu[..., 2]
    mat[..., 2, 1] = triu[..., 4]
    mat[..., 2, 2] = triu[..., 5]
    return mat


# ---------------------------------------------------------------------------
# create_camera_model
# ---------------------------------------------------------------------------


def create_camera_model(
    camera_model: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    principal_points: Optional[Tensor] = None,
    focal_lengths: Optional[Tensor] = None,
    radial_coeffs: Optional[Tensor] = None,
    tangential_coeffs: Optional[Tensor] = None,
    thin_prism_coeffs: Optional[Tensor] = None,
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
    external_distortion_coeffs: Optional[BivariateWindshieldModelParameters] = None,
    rs_type: RollingShutterType = RollingShutterType.GLOBAL,
    lidar_coeffs: Optional["RowOffsetStructuredSpinningLidarModelParametersExt"] = None,
):
    """Create a camera model (pure-Python path for MPS)."""
    from gsplat.cuda._torch_cameras import _BaseCameraModel

    if camera_model == "lidar":
        assert (
            lidar_coeffs is not None
        ), "lidar_coeffs is required for lidar camera model"
        return _BaseCameraModel.create(
            camera_model=camera_model,
            lidar_coeffs=lidar_coeffs,
        )
    else:
        assert width is not None, "width is required for non-lidar camera models"
        assert height is not None, "height is required for non-lidar camera models"
        assert (
            principal_points is not None
        ), "principal_points is required for non-lidar camera models"
        return _BaseCameraModel.create(
            width=width,
            height=height,
            camera_model=camera_model,
            principal_points=principal_points,
            focal_lengths=focal_lengths,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs,
            ftheta_coeffs=ftheta_coeffs,
            rs_type=rs_type,
        )


# ===================================================================
# PUBLIC API FUNCTIONS
# ===================================================================


def world_to_cam(
    means: Tensor,  # [..., N, 3]
    covars: Tensor,  # [..., N, 3, 3]
    viewmats: Tensor,  # [..., C, 4, 4]
) -> Tuple[Tensor, Tensor]:
    """Transforms Gaussians from world to camera coordinate system.

    Args:
        means: Gaussian means. [..., N, 3]
        covars: Gaussian covariances. [..., N, 3, 3]
        viewmats: World-to-camera transformation matrices. [..., C, 4, 4]

    Returns:
        A tuple:

        - **Gaussian means in camera coordinate system**. [..., C, N, 3]
        - **Gaussian covariances in camera coordinate system**. [..., C, N, 3, 3]
    """
    from gsplat.cuda._torch_impl import _world_to_cam

    warnings.warn(
        "world_to_cam() is removed from the CUDA backend as it's relatively easy to "
        "implement in PyTorch. Currently use the PyTorch implementation instead. "
        "This function will be completely removed in a future release.",
        DeprecationWarning,
    )
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    assert means.shape == batch_dims + (N, 3), means.shape
    assert covars.shape == batch_dims + (N, 3, 3), covars.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    means = means.contiguous()
    covars = covars.contiguous()
    viewmats = viewmats.contiguous()
    return _world_to_cam(means, covars, viewmats)


def adam(
    param: Tensor,
    param_grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    valid: Tensor,
    lr: float,
    b1: float,
    b2: float,
    eps: float,
) -> None:
    """Pure-PyTorch Adam step (replaces the CUDA kernel).

    .. todo:: MPS: Replace with a fused Metal kernel for better performance.
        CUDA equivalent: ``AdamCUDA.cu``
    """
    exp_avg.mul_(b1).add_(param_grad, alpha=1 - b1)
    exp_avg_sq.mul_(b2).addcmul_(param_grad, param_grad, value=1 - b2)
    step = exp_avg / (exp_avg_sq.sqrt() + eps)
    param.data[valid] -= lr * step[valid]


def spherical_harmonics(
    degrees_to_use: int,
    dirs: Tensor,  # [..., 3]
    coeffs: Tensor,  # [..., K, 3]
    masks: Optional[Tensor] = None,  # [...,]
) -> Tensor:
    """Computes spherical harmonics.

    Args:
        degrees_to_use: The degree to be used.
        dirs: Directions. [..., 3]
        coeffs: Coefficients. [..., K, 3]
        masks: Optional boolen masks to skip some computation. [...,] Default: None.

    Returns:
        Spherical harmonics. [..., 3]
    """
    from gsplat.cuda._torch_impl import _spherical_harmonics

    assert (degrees_to_use + 1) ** 2 <= coeffs.shape[-2], coeffs.shape
    batch_dims = dirs.shape[:-1]
    assert dirs.shape == batch_dims + (3,), dirs.shape
    assert (
        (len(coeffs.shape) == len(batch_dims) + 2)
        and coeffs.shape[:-2] == batch_dims
        and coeffs.shape[-1] == 3
    ), coeffs.shape
    if masks is not None:
        assert masks.shape == batch_dims, masks.shape

    if _mps_backend_available() and dirs.device.type == "mps":
        N = int(dirs.numel() // 3)
        K = int(coeffs.shape[-2])
        dirs_flat = dirs.contiguous().reshape(N, 3)
        coeffs_flat = coeffs.contiguous().reshape(N, K, 3)
        masks_flat = (
            masks.contiguous().reshape(N).to(torch.bool)
            if masks is not None
            else None
        )
        colors_flat = _SphericalHarmonics.apply(
            dirs_flat, coeffs_flat, masks_flat, K, int(degrees_to_use)
        )
        return colors_flat.reshape(batch_dims + (3,))

    # TODO: MPS: The pure-PyTorch reference does not support masks; callers
    # that pass a mask on non-MPS devices will see it silently ignored.
    return _spherical_harmonics(
        degrees_to_use, dirs.contiguous(), coeffs.contiguous()
    )


def quat_scale_to_covar_preci(
    quats: Tensor,  # [..., 4],
    scales: Tensor,  # [..., 3],
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Converts quaternions and scales to covariance and precision matrices.

    Args:
        quats: Quaternions (No need to be normalized). [..., 4]
        scales: Scales. [..., 3]
        compute_covar: Whether to compute covariance matrices. Default: True.
        compute_preci: Whether to compute precision matrices. Default: True.
        triu: If True, the return matrices will be upper triangular. Default: False.

    Returns:
        A tuple:

        - **Covariance matrices**. If `triu` is True [..., 6], otherwise [..., 3, 3].
        - **Precision matrices**. If `triu` is True [..., 6], otherwise [..., 3, 3].
    """
    batch_dims = quats.shape[:-1]
    assert quats.shape == batch_dims + (4,), quats.shape
    assert scales.shape == batch_dims + (3,), scales.shape

    # Native Metal path when the MPS backend compiled. Fall back to the pure-
    # PyTorch oracle otherwise (e.g. CPU-only dev boxes, torch builds without
    # torch.mps.compile_shader).
    if _mps_backend_available() and quats.device.type == "mps":
        quats_c = quats.contiguous()
        scales_c = scales.contiguous()
        covars_flat, precis_flat = _QuatScaleToCovarPreci.apply(
            quats_c.reshape(-1, 4),
            scales_c.reshape(-1, 3),
            compute_covar,
            compute_preci,
            triu,
        )
        out_shape = batch_dims + ((6,) if triu else (3, 3))
        covars = covars_flat.reshape(out_shape) if compute_covar else None
        precis = precis_flat.reshape(out_shape) if compute_preci else None
        return covars, precis

    from gsplat.cuda._math import _quat_scale_to_covar_preci

    covars, precis = _quat_scale_to_covar_preci(
        quats.contiguous(), scales.contiguous(), compute_covar, compute_preci, triu
    )
    return covars if compute_covar else None, precis if compute_preci else None


def _mps_backend_available() -> bool:
    try:
        from ._backend import _C
    except Exception:
        return False
    return _C is not None


class _QuatScaleToCovarPreci(torch.autograd.Function):
    """Native Metal forward/backward for quat_scale_to_covar_preci."""

    @staticmethod
    def forward(
        ctx,
        quats: Tensor,  # [N, 4] contiguous, float32, device=mps
        scales: Tensor,  # [N, 3] contiguous, float32, device=mps
        compute_covar: bool,
        compute_preci: bool,
        triu: bool,
    ) -> Tuple[Tensor, Tensor]:
        N = quats.shape[0]
        assert quats.shape == (N, 4)
        assert scales.shape == (N, 3)
        assert quats.device.type == "mps" and scales.device.type == "mps"

        out_dim = 6 if triu else 9
        # One of the two outputs may be unused; allocate a 1-element sentinel
        # when not requested so the kernel always has a valid buffer binding.
        if compute_covar:
            covars_flat = torch.empty(N * out_dim, dtype=quats.dtype, device=quats.device)
        else:
            covars_flat = torch.empty(1, dtype=quats.dtype, device=quats.device)
        if compute_preci:
            precis_flat = torch.empty(N * out_dim, dtype=quats.dtype, device=quats.device)
        else:
            precis_flat = torch.empty(1, dtype=quats.dtype, device=quats.device)

        fwd = _get_mps_backend().get_kernel("gsplat_quat_scale_to_covar_preci_fwd")
        fwd(
            quats,
            scales,
            covars_flat,
            precis_flat,
            int(N),
            1 if triu else 0,
            1 if compute_covar else 0,
            1 if compute_preci else 0,
            threads=int(N),
        )

        ctx.save_for_backward(quats, scales)
        ctx.compute_covar = compute_covar
        ctx.compute_preci = compute_preci
        ctx.triu = triu

        out_shape = (N, 6) if triu else (N, 3, 3)
        covars = covars_flat.reshape(out_shape) if compute_covar else covars_flat[:0]
        precis = precis_flat.reshape(out_shape) if compute_preci else precis_flat[:0]
        return covars, precis

    @staticmethod
    def backward(ctx, v_covars: Tensor, v_precis: Tensor):
        quats, scales = ctx.saved_tensors
        compute_covar: bool = ctx.compute_covar
        compute_preci: bool = ctx.compute_preci
        triu: bool = ctx.triu

        N = quats.shape[0]
        out_dim = 6 if triu else 9

        if compute_covar:
            if v_covars.is_sparse:
                v_covars = v_covars.to_dense()
            v_covars_flat = v_covars.contiguous().reshape(N * out_dim)
        else:
            v_covars_flat = torch.empty(1, dtype=quats.dtype, device=quats.device)

        if compute_preci:
            if v_precis.is_sparse:
                v_precis = v_precis.to_dense()
            v_precis_flat = v_precis.contiguous().reshape(N * out_dim)
        else:
            v_precis_flat = torch.empty(1, dtype=quats.dtype, device=quats.device)

        v_quats = torch.empty(N * 4, dtype=quats.dtype, device=quats.device)
        v_scales = torch.empty(N * 3, dtype=quats.dtype, device=quats.device)

        bwd = _get_mps_backend().get_kernel("gsplat_quat_scale_to_covar_preci_bwd")
        bwd(
            quats,
            scales,
            v_covars_flat,
            v_precis_flat,
            v_quats,
            v_scales,
            int(N),
            1 if triu else 0,
            1 if compute_covar else 0,
            1 if compute_preci else 0,
            threads=int(N),
        )

        return (
            v_quats.reshape(N, 4),
            v_scales.reshape(N, 3),
            None,  # compute_covar
            None,  # compute_preci
            None,  # triu
        )


class _SphericalHarmonics(torch.autograd.Function):
    """Native Metal forward/backward for spherical_harmonics."""

    @staticmethod
    def forward(
        ctx,
        dirs: Tensor,  # [N, 3] contiguous, float32, device=mps
        coeffs: Tensor,  # [N, K, 3] contiguous, float32, device=mps
        masks: Optional[Tensor],  # [N] bool, device=mps, or None
        K: int,
        degrees_to_use: int,
    ) -> Tensor:
        N = dirs.shape[0]
        assert dirs.shape == (N, 3)
        assert coeffs.shape == (N, K, 3)
        assert dirs.device.type == "mps" and coeffs.device.type == "mps"

        has_mask = masks is not None
        if has_mask:
            assert masks.shape == (N,) and masks.dtype == torch.bool
            masks_u8 = masks.to(torch.uint8).contiguous()
        else:
            # Sentinel — never read because has_mask=0.
            masks_u8 = torch.zeros(1, dtype=torch.uint8, device=dirs.device)

        # Pre-zeroed so masked rows (and unused higher bands) stay at 0.
        colors_flat = torch.zeros(N * 3, dtype=dirs.dtype, device=dirs.device)

        fwd = _get_mps_backend().get_kernel("gsplat_spherical_harmonics_fwd")
        fwd(
            dirs,
            coeffs,
            masks_u8,
            colors_flat,
            int(N),
            int(K),
            int(degrees_to_use),
            1 if has_mask else 0,
            threads=int(N),
        )

        ctx.save_for_backward(dirs, coeffs, masks_u8)
        ctx.N = N
        ctx.K = K
        ctx.degrees_to_use = degrees_to_use
        ctx.has_mask = has_mask
        return colors_flat.reshape(N, 3)

    @staticmethod
    def backward(ctx, v_colors: Tensor):
        dirs, coeffs, masks_u8 = ctx.saved_tensors
        N: int = ctx.N
        K: int = ctx.K
        degrees_to_use: int = ctx.degrees_to_use
        has_mask: bool = ctx.has_mask

        v_colors_flat = v_colors.contiguous().reshape(N * 3)

        # Pre-zero outputs so masked rows (and unused higher SH bands) remain
        # at zero — the kernel returns early without writing them.
        v_coeffs_flat = torch.zeros(N * K * 3, dtype=dirs.dtype, device=dirs.device)

        compute_v_dirs = ctx.needs_input_grad[0]
        if compute_v_dirs:
            v_dirs_flat = torch.zeros(N * 3, dtype=dirs.dtype, device=dirs.device)
        else:
            v_dirs_flat = torch.zeros(1, dtype=dirs.dtype, device=dirs.device)

        bwd = _get_mps_backend().get_kernel("gsplat_spherical_harmonics_bwd")
        bwd(
            dirs,
            coeffs,
            masks_u8,
            v_colors_flat,
            v_coeffs_flat,
            v_dirs_flat,
            int(N),
            int(K),
            int(degrees_to_use),
            1 if has_mask else 0,
            1 if compute_v_dirs else 0,
            threads=int(N),
        )

        v_dirs = v_dirs_flat.reshape(N, 3) if compute_v_dirs else None
        v_coeffs = v_coeffs_flat.reshape(N, K, 3)
        return (
            v_dirs,    # dirs
            v_coeffs,  # coeffs
            None,      # masks
            None,      # K
            None,      # degrees_to_use
        )


def persp_proj(
    means: Tensor,  # [..., C, N, 3]
    covars: Tensor,  # [..., C, N, 3, 3]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """Perspective projection on Gaussians.
    DEPRECATED: please use `proj` with `ortho=False` instead.

    Args:
        means: Gaussian means. [..., C, N, 3]
        covars: Gaussian covariances. [..., C, N, 3, 3]
        Ks: Camera intrinsics. [..., C, 3, 3]
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **Projected means**. [..., C, N, 2]
        - **Projected covariances**. [..., C, N, 2, 2]
    """
    warnings.warn(
        "persp_proj is deprecated and will be removed in a future release. "
        "Use proj with ortho=False instead.",
        DeprecationWarning,
    )
    return proj(means, covars, Ks, width, height, camera_model="pinhole")


def proj(
    means: Tensor,  # [..., C, N, 3]
    covars: Tensor,  # [..., C, N, 3, 3]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    camera_model: CameraModel = "pinhole",
) -> Tuple[Tensor, Tensor]:
    """Projection of Gaussians (perspective, orthographic, or fisheye).

    Args:
        means: Gaussian means. [..., C, N, 3]
        covars: Gaussian covariances. [..., C, N, 3, 3]
        Ks: Camera intrinsics. [..., C, 3, 3]
        width: Image width.
        height: Image height.
        camera_model: Camera model. Default: "pinhole".

    Returns:
        A tuple:

        - **Projected means**. [..., C, N, 2]
        - **Projected covariances**. [..., C, N, 2, 2]
    """
    from gsplat.cuda._torch_impl import _persp_proj, _fisheye_proj, _ortho_proj

    assert (
        camera_model != "ftheta"
    ), "ftheta camera is only supported via UT, please set with_ut=True in the rasterization()"

    batch_dims = means.shape[:-3]
    C, N = means.shape[-3:-1]
    assert means.shape == batch_dims + (C, N, 3), means.shape
    assert covars.shape == batch_dims + (C, N, 3, 3), covars.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    means = means.contiguous()
    covars = covars.contiguous()
    Ks = Ks.contiguous()

    # TODO: MPS: Replace with a Metal kernel for better performance.
    #   CUDA equivalent: ``ProjectionEWASimple.cu``
    if camera_model == "pinhole":
        return _persp_proj(means, covars, Ks, width, height)
    elif camera_model == "fisheye":
        return _fisheye_proj(means, covars, Ks, width, height)
    elif camera_model == "ortho":
        return _ortho_proj(means, covars, Ks, width, height)
    else:
        raise ValueError(f"Unsupported camera model for proj(): {camera_model}")


def fully_fused_projection(
    means: Tensor,  # [..., N, 3]
    covars: Optional[Tensor],  # [..., N, 6] or None
    quats: Optional[Tensor],  # [..., N, 4] or None
    scales: Optional[Tensor],  # [..., N, 3] or None
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
    calc_compensations: bool = False,
    camera_model: CameraModel = "pinhole",
    opacities: Optional[Tensor] = None,  # [..., N] or None
) -> Tuple[Tensor, ...]:
    """Projects Gaussians to 2D.

    Delegates to ``_torch_impl._fully_fused_projection`` which expects full
    ``[..., N, 3, 3]`` covariance matrices.  When the caller supplies the
    upper-triangle packed format ``[..., N, 6]`` (or quaternions + scales)
    we convert first.

    .. todo:: MPS: Replace with a fused Metal kernel.
        CUDA equivalents: ``ProjectionEWA3DGSFused.cu`` (dense),
        ``ProjectionEWA3DGSPacked.cu`` (packed).
    """
    from gsplat.cuda._torch_impl import _fully_fused_projection
    from gsplat.cuda._math import _quat_scale_to_covar_preci as _qs2cp

    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    assert means.shape == batch_dims + (N, 3), means.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    means = means.contiguous()

    if covars is not None:
        assert covars.shape == batch_dims + (N, 6), covars.shape
        covars = covars.contiguous()
        # Convert upper-triangle [..., N, 6] -> full [..., N, 3, 3]
        covars_3x3 = _triu_to_full(covars)
    else:
        assert quats is not None, "covars or quats is required"
        assert scales is not None, "covars or scales is required"
        assert quats.shape == batch_dims + (N, 4), quats.shape
        assert scales.shape == batch_dims + (N, 3), scales.shape
        quats = quats.contiguous()
        scales = scales.contiguous()
        # Compute covariance from quats + scales (triu format) then convert.
        covars_triu, _ = _qs2cp(quats, scales, compute_covar=True, compute_preci=False, triu=True)
        covars_3x3 = _triu_to_full(covars_triu)

    if sparse_grad:
        assert packed, "sparse_grad is only supported when packed is True"
        assert batch_dims == (), "sparse_grad does not support batch dimensions"
    if opacities is not None:
        assert opacities.shape == batch_dims + (N,), opacities.shape
        opacities = opacities.contiguous()

    assert (
        camera_model != "ftheta"
    ), "ftheta camera is only supported via UT, please set with_ut=True in the rasterization()"

    viewmats = viewmats.contiguous()
    Ks = Ks.contiguous()

    radii, means2d, depths, conics, compensations = _fully_fused_projection(
        means,
        covars_3x3,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=far_plane,
        calc_compensations=calc_compensations,
        camera_model=camera_model,
    )

    if not packed:
        return radii, means2d, depths, conics, compensations

    # --- packed mode: flatten valid entries into COO-style tensors ---
    # radii: [..., C, N, 2], valid where any component > 0
    valid = (radii > 0).any(dim=-1)  # [..., C, N]
    # Flatten batch dims into B
    B = math.prod(batch_dims)
    radii_flat = radii.reshape(B, C, N, 2)
    means2d_flat = means2d.reshape(B, C, N, 2)
    depths_flat = depths.reshape(B, C, N)
    conics_flat = conics.reshape(B, C, N, 3)
    valid_flat = valid.reshape(B, C, N)
    if compensations is not None:
        comp_flat = compensations.reshape(B, C, N)

    # Compute per batch-camera indptr
    counts_per_bc = valid_flat.reshape(B * C, N).sum(dim=-1)  # [B*C]
    indptr = torch.zeros(B * C + 1, dtype=torch.int32, device=means.device)
    torch.cumsum(counts_per_bc, dim=0, out=indptr[1:])

    # Gather valid indices
    batch_idx, camera_idx, gauss_idx = torch.where(valid_flat)
    batch_ids = batch_idx.int()
    camera_ids = camera_idx.int()
    gaussian_ids = gauss_idx.int()

    # Gather packed tensors
    packed_radii = radii_flat[batch_idx, camera_idx, gauss_idx]
    packed_means2d = means2d_flat[batch_idx, camera_idx, gauss_idx]
    packed_depths = depths_flat[batch_idx, camera_idx, gauss_idx]
    packed_conics = conics_flat[batch_idx, camera_idx, gauss_idx]
    packed_compensations = (
        comp_flat[batch_idx, camera_idx, gauss_idx]
        if compensations is not None
        else None
    )

    return (
        batch_ids,
        camera_ids,
        gaussian_ids,
        indptr,
        packed_radii,
        packed_means2d,
        packed_depths,
        packed_conics,
        packed_compensations,
    )


@torch.no_grad()
def isect_tiles(
    means2d: Tensor,  # [..., N, 2] or [nnz, 2]
    radii: Tensor,  # [..., N, 2] or [nnz, 2]
    depths: Tensor,  # [..., N] or [nnz]
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
    segmented: bool = False,
    packed: bool = False,
    n_images: Optional[int] = None,
    image_ids: Optional[Tensor] = None,
    gaussian_ids: Optional[Tensor] = None,
    conics: Optional[Tensor] = None,  # [..., N, 3] or [nnz, 3]
    opacities: Optional[Tensor] = None,  # [..., N] or [nnz]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Maps projected Gaussians to intersecting tiles.

    Delegates to ``_torch_impl._isect_tiles``.  The AccuTile path
    (``conics`` / ``opacities``) and packed mode are **not** supported by the
    pure-PyTorch reference; they are silently ignored.

    .. todo:: MPS: Replace with a Metal kernel for better performance.
        Also implement AccuTile conservative ellipse intersection.
        CUDA equivalent: ``IntersectTile.cu``
    """
    from gsplat.cuda._torch_impl import _isect_tiles

    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert radii.shape == (nnz, 2), radii.shape
        assert depths.shape == (nnz,), depths.shape
        assert image_ids is not None, "image_ids is required if packed is True"
        assert gaussian_ids is not None, "gaussian_ids is required if packed is True"
        assert n_images is not None, "n_images is required if packed is True"

        # Unpack into dense format so we can reuse the dense _isect_tiles.
        # Determine max N across all images.
        N = int(gaussian_ids.max().item()) + 1 if nnz > 0 else 0
        I = n_images
        device = means2d.device

        dense_means2d = torch.zeros(I, N, 2, device=device, dtype=means2d.dtype)
        dense_radii = torch.zeros(I, N, 2, device=device, dtype=radii.dtype)
        dense_depths = torch.zeros(I, N, device=device, dtype=depths.dtype)

        dense_means2d[image_ids.long(), gaussian_ids.long()] = means2d
        dense_radii[image_ids.long(), gaussian_ids.long()] = radii
        dense_depths[image_ids.long(), gaussian_ids.long()] = depths

        tiles_per_gauss, isect_ids, flatten_ids = _isect_tiles(
            dense_means2d, dense_radii, dense_depths, tile_size, tile_width, tile_height, sort
        )
        return tiles_per_gauss, isect_ids, flatten_ids
    else:
        image_dims = means2d.shape[:-2]
        N = means2d.shape[-2]
        assert means2d.shape == image_dims + (N, 2), means2d.shape
        assert radii.shape == image_dims + (N, 2), radii.shape
        assert depths.shape == image_dims + (N,), depths.shape

    return _isect_tiles(
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        tile_size,
        tile_width,
        tile_height,
        sort,
    )


@torch.no_grad()
def isect_tiles_lidar(
    lidar: RowOffsetStructuredSpinningLidarModelParametersExt,
    means2d: Tensor,  # [..., N, 2] or [nnz, 2]
    radii: Tensor,  # [..., N, 2] or [nnz, 2]
    depths: Tensor,  # [..., N] or [nnz]
    sort: bool = True,
    segmented: bool = False,
    packed: bool = False,
    n_images: Optional[int] = None,
    image_ids: Optional[Tensor] = None,
    gaussian_ids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Maps projected Gaussians to intersecting tiles (lidar variant).

    Not yet implemented for MPS — raises ``NotImplementedError``.
    """
    # TODO: MPS: Implement lidar tile intersection as a Metal kernel.
    #   CUDA equivalent: ``IntersectTileLidar.cu``
    raise NotImplementedError(
        "isect_tiles_lidar is not yet supported on the MPS backend. "
        "It requires a native Metal kernel for lidar tile intersection."
    )


@torch.no_grad()
def isect_offset_encode(
    isect_ids: Tensor,
    n_images: int,
    tile_width: int,
    tile_height: int,
) -> Tensor:
    """Encodes intersection ids to offsets.

    Args:
        isect_ids: Intersection ids. [n_isects]
        n_images: Number of images.
        tile_width: Tile width.
        tile_height: Tile height.

    Returns:
        Offsets. [I, tile_height, tile_width]
    """
    # TODO: MPS: Replace with a Metal kernel for better performance.
    #   CUDA equivalent: part of ``IntersectTile.cu``
    from gsplat.cuda._torch_impl import _isect_offset_encode

    return _isect_offset_encode(
        isect_ids.contiguous(), n_images, tile_width, tile_height
    )


def rasterize_to_pixels(
    means2d: Tensor,  # [..., N, 2] or [nnz, 2]
    conics: Tensor,  # [..., N, 3] or [nnz, 3]
    colors: Tensor,  # [..., N, channels] or [nnz, channels]
    opacities: Tensor,  # [..., N] or [nnz]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [..., channels]
    masks: Optional[Tensor] = None,  # [..., tile_height, tile_width]
    packed: bool = False,
    absgrad: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Rasterizes Gaussians to pixels.

    Delegates to ``_torch_impl._rasterize_to_pixels``.

    .. todo:: MPS: Replace with a fused Metal kernel for forward + backward.
        CUDA equivalents: ``RasterizeToPixels3DGSFwd.cu``,
        ``RasterizeToPixels3DGSBwd.cu``

    Returns:
        A tuple:

        - **Rendered colors**. [..., image_height, image_width, channels]
        - **Rendered alphas**. [..., image_height, image_width, 1]
    """
    from gsplat.cuda._torch_impl import _rasterize_to_pixels

    image_dims = means2d.shape[:-2]
    channels = colors.shape[-1]
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert conics.shape == (nnz, 3), conics.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(-2)
        assert means2d.shape == image_dims + (N, 2), means2d.shape
        assert conics.shape == image_dims + (N, 3), conics.shape
        assert colors.shape == image_dims + (N, channels), colors.shape
        assert opacities.shape == image_dims + (N,), opacities.shape
    if backgrounds is not None:
        assert backgrounds.shape == image_dims + (channels,), backgrounds.shape
        backgrounds = backgrounds.contiguous()
    if masks is not None:
        assert masks.shape == isect_offsets.shape, masks.shape
        masks = masks.contiguous()

    # Channel padding logic (mirrors CUDA wrapper)
    if channels > 513 or channels == 0:
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (
        1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65,
        128, 129, 256, 257, 512, 513,
    ):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [
                colors,
                torch.zeros(*colors.shape[:-1], padded_channels, device=device),
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0

    tile_height, tile_width = isect_offsets.shape[-2:]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    render_colors, render_alphas = _rasterize_to_pixels(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        backgrounds,
    )

    if padded_channels > 0:
        render_colors = render_colors[..., :-padded_channels]
    return render_colors, render_alphas


def rasterize_to_pixels_eval3d(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    colors: Tensor,  # [..., C, N, channels] or [nnz, channels]
    opacities: Tensor,  # [..., C, N] or [nnz]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [..., C, channels]
    masks: Optional[Tensor] = None,  # [..., C, tile_height, tile_width]
    camera_model: CameraModel = "pinhole",
    ut_params: Optional[UnscentedTransformParameters] = None,
    rays: Optional[Tensor] = None,  # [..., C, H, W, 6]
    # distortion
    radial_coeffs: Optional[Tensor] = None,  # [..., C, 6] or [..., C, 4]
    tangential_coeffs: Optional[Tensor] = None,  # [..., C, 2]
    thin_prism_coeffs: Optional[Tensor] = None,  # [..., C, 4]
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
    lidar_coeffs: Optional[RowOffsetStructuredSpinningLidarModelParametersExt] = None,
    external_distortion_coeffs: Optional[BivariateWindshieldModelParameters] = None,
    # rolling shutter
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: Optional[Tensor] = None,  # [..., C, 4, 4]
    use_hit_distance: bool = False,
    return_normals: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Rasterizes Gaussians to pixels (eval3d path).

    Returns:
        A tuple:

        - **Rendered colors**. [..., C, image_height, image_width, channels]
        - **Rendered alphas**. [..., C, image_height, image_width, 1]
    """
    if ut_params is None:
        ut_params = UnscentedTransformParameters()

    colors, alphas, *_ = rasterize_to_pixels_eval3d_extra(
        means=means,
        quats=quats,
        scales=scales,
        colors=colors,
        opacities=opacities,
        viewmats=viewmats,
        Ks=Ks,
        rays=rays,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        backgrounds=backgrounds,
        masks=masks,
        camera_model=camera_model,
        ut_params=ut_params,
        radial_coeffs=radial_coeffs,
        tangential_coeffs=tangential_coeffs,
        thin_prism_coeffs=thin_prism_coeffs,
        ftheta_coeffs=ftheta_coeffs,
        lidar_coeffs=lidar_coeffs,
        external_distortion_coeffs=external_distortion_coeffs,
        rolling_shutter=rolling_shutter,
        viewmats_rs=viewmats_rs,
        return_sample_counts=False,
        use_hit_distance=use_hit_distance,
        return_normals=return_normals,
    )
    return colors, alphas


def rasterize_to_pixels_eval3d_extra(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    colors: Tensor,  # [..., C, N, channels] or [nnz, channels]
    opacities: Tensor,  # [..., C, N] or [nnz]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [..., C, channels]
    masks: Optional[Tensor] = None,  # [..., C, tile_height, tile_width]
    camera_model: CameraModel = "pinhole",
    ut_params: Optional[UnscentedTransformParameters] = None,
    rays: Optional[Tensor] = None,  # [..., C, P, 6]
    # distortion
    radial_coeffs: Optional[Tensor] = None,  # [..., C, 6] or [..., C, 4]
    tangential_coeffs: Optional[Tensor] = None,  # [..., C, 2]
    thin_prism_coeffs: Optional[Tensor] = None,  # [..., C, 4]
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
    lidar_coeffs: Optional[RowOffsetStructuredSpinningLidarModelParametersExt] = None,
    external_distortion_coeffs: Optional[BivariateWindshieldModelParameters] = None,
    # rolling shutter
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: Optional[Tensor] = None,  # [..., C, 4, 4]
    return_sample_counts: bool = False,
    use_hit_distance: bool = False,
    return_normals: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    """Rasterizes Gaussians to pixels, returning extra information for debugging.

    Delegates to ``_torch_impl_eval3d._rasterize_to_pixels_eval3d``.

    .. todo:: MPS: Replace with fused Metal kernels for forward + backward.
        CUDA equivalents: ``RasterizeToPixelsFromWorld3DGSFwd.cu``,
        ``RasterizeToPixelsFromWorld3DGSBwd.cu``

    Returns:
        A tuple:

        - **Rendered colors**. [..., C, image_height, image_width, channels]
        - **Rendered alphas**. [..., C, image_height, image_width, 1]
        - **Last flatten_idx**. [..., C, image_height, image_width]
        - **Sample counts** (optional). [..., C, image_height, image_width].
        - **Rendered normals** (optional). [..., C, image_height, image_width, 3].
    """
    from gsplat.cuda._torch_impl_eval3d import _rasterize_to_pixels_eval3d

    if ut_params is None:
        ut_params = UnscentedTransformParameters()

    batch_dims = means.shape[:-2]
    num_batch_dims = len(batch_dims)
    N = means.size(-2)
    C = viewmats.size(-3)
    P = rays.shape[-2] if rays is not None else 0
    channels = colors.shape[-1]
    device = means.device

    assert means.shape == batch_dims + (N, 3), means.shape
    assert quats.shape == batch_dims + (N, 4), quats.shape
    assert scales.shape == batch_dims + (N, 3), scales.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    if rays is not None:
        assert_shape("rays", rays, batch_dims + (C, P, 6))

    assert colors.ndim in (num_batch_dims + 2, num_batch_dims + 3), colors.shape
    if colors.ndim == num_batch_dims + 2:
        raise NotImplementedError("packed mode is not supported yet")
    else:
        assert colors.shape == batch_dims + (C, N, channels), colors.shape
    assert opacities.shape == colors.shape[:-1], opacities.shape

    if backgrounds is not None:
        assert backgrounds.shape == batch_dims + (C, channels), backgrounds.shape
        backgrounds = backgrounds.contiguous()
    if masks is not None:
        assert masks.shape == isect_offsets.shape, masks.shape
        masks = masks.contiguous()
    if radial_coeffs is not None:
        assert radial_coeffs.shape[:-1] == batch_dims + (C,) and radial_coeffs.shape[
            -1
        ] in (6, 4), radial_coeffs.shape
        radial_coeffs = radial_coeffs.contiguous()
    if tangential_coeffs is not None:
        assert tangential_coeffs.shape == batch_dims + (C, 2), tangential_coeffs.shape
        tangential_coeffs = tangential_coeffs.contiguous()
    if thin_prism_coeffs is not None:
        assert thin_prism_coeffs.shape == batch_dims + (C, 4), thin_prism_coeffs.shape
        thin_prism_coeffs = thin_prism_coeffs.contiguous()
    if viewmats_rs is not None:
        assert viewmats_rs.shape == batch_dims + (C, 4, 4), viewmats_rs.shape
        viewmats_rs = viewmats_rs.contiguous()

    # Channel padding (mirrors CUDA wrapper)
    channels = colors.shape[-1]
    if channels > 513 or channels == 0:
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (
        1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65,
        128, 129, 256, 257, 512, 513,
    ):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [
                colors[..., :-1],
                torch.zeros(*colors.shape[:-1], padded_channels, device=device),
                colors[..., -1:],
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0

    tile_height, tile_width = isect_offsets.shape[-2:]
    if camera_model == "lidar":
        assert lidar_coeffs is not None
        assert tile_width == lidar_coeffs.tiling.n_bins_azimuth
        assert tile_height == lidar_coeffs.tiling.n_bins_elevation
    else:
        assert (
            tile_height * tile_size >= image_height
        ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
        assert (
            tile_width * tile_size >= image_width
        ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    result = _rasterize_to_pixels_eval3d(
        means=means.contiguous(),
        quats=quats.contiguous(),
        scales=scales.contiguous(),
        colors=colors.contiguous(),
        opacities=opacities.contiguous(),
        viewmats=viewmats.contiguous(),
        Ks=Ks.contiguous(),
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        isect_offsets=isect_offsets.contiguous(),
        flatten_ids=flatten_ids.contiguous(),
        backgrounds=backgrounds,
        rs_type=rolling_shutter,
        rays=rays.contiguous() if rays is not None else None,
        viewmats_rs=viewmats_rs,
        return_last_ids=True,
        return_sample_counts=return_sample_counts,
        use_hit_distance=use_hit_distance,
        return_normals=return_normals,
        lidar_coeffs=lidar_coeffs,
    )

    # _rasterize_to_pixels_eval3d returns a variable-length tuple depending on
    # what was requested.  Normalise into the fixed 5-element return expected
    # by the public API.
    render_colors = result[0]
    render_alphas = result[1]

    # Parse optional outputs from the reference implementation
    idx = 2
    last_ids = result[idx] if idx < len(result) else torch.zeros(
        batch_dims + (C, image_height, image_width),
        dtype=torch.int32, device=device,
    )
    idx += 1
    sample_counts = result[idx] if return_sample_counts and idx < len(result) else None
    if return_sample_counts:
        idx += 1
    render_normals = result[idx] if return_normals and idx < len(result) else None

    if padded_channels > 0:
        render_colors = torch.cat(
            [render_colors[..., : -padded_channels - 1], render_colors[..., -1:]],
            dim=-1,
        )

    return render_colors, render_alphas, last_ids, sample_counts, render_normals


@torch.no_grad()
def rasterize_to_indices_in_range(
    range_start: int,
    range_end: int,
    transmittances: Tensor,  # [..., image_height, image_width]
    means2d: Tensor,  # [..., N, 2]
    conics: Tensor,  # [..., N, 3]
    opacities: Tensor,  # [..., N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pure-PyTorch implementation of ``rasterize_to_indices_in_range``.

    This is a reference (slow, loop-based) implementation that produces the
    same output as the CUDA kernel.  It iterates over tiles and pixels to
    find Gaussian-pixel intersections within the requested batch range.

    .. todo:: MPS: Replace with a Metal kernel for better performance.
        This is the single biggest performance bottleneck in the pure-PyTorch
        rasterization path. CUDA equivalent: ``RasterizeToIndices3DGS.cu``

    Returns:
        A tuple:

        - **Gaussian ids**. [M]
        - **Pixel ids**. [M]
        - **Image ids**. [M]
    """
    from gsplat.cuda._constants import ALPHA_THRESHOLD, TRANSMITTANCE_THRESHOLD

    image_dims = means2d.shape[:-2]
    tile_height, tile_width = isect_offsets.shape[-2:]
    N = means2d.shape[-2]
    assert transmittances.shape == image_dims + (
        image_height,
        image_width,
    ), transmittances.shape
    assert means2d.shape == image_dims + (N, 2), means2d.shape
    assert conics.shape == image_dims + (N, 3), conics.shape
    assert opacities.shape == image_dims + (N,), opacities.shape
    assert isect_offsets.shape == image_dims + (
        tile_height,
        tile_width,
    ), isect_offsets.shape
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    I = math.prod(image_dims)
    device = means2d.device

    means2d_flat = means2d.reshape(I * N, 2)
    conics_flat = conics.reshape(I * N, 3)
    opacities_flat = opacities.reshape(I * N)
    # offsets_flat[t] = start index for tile t in flatten_ids.
    # The end index is offsets_flat[t+1] (or n_isects for the last tile).
    offsets_flat = isect_offsets.reshape(I * tile_height * tile_width)
    n_isects = flatten_ids.shape[0]
    # Build an extended offsets array with a sentinel at the end.
    offsets_ext = torch.cat(
        [offsets_flat, torch.tensor([n_isects], device=offsets_flat.device, dtype=offsets_flat.dtype)]
    )
    trans_flat = transmittances.reshape(I, image_height, image_width)

    block_size = tile_size * tile_size

    out_gauss_ids_list = []
    out_pixel_ids_list = []
    out_image_ids_list = []

    for img_id in range(I):
        for ty in range(tile_height):
            for tx in range(tile_width):
                tile_flat_idx = img_id * tile_height * tile_width + ty * tile_width + tx
                tile_start = offsets_ext[tile_flat_idx].item()
                tile_end = offsets_ext[tile_flat_idx + 1].item()

                n_gauss_in_tile = tile_end - tile_start
                if n_gauss_in_tile == 0:
                    continue

                # Determine the gaussian index range within this tile for the
                # requested batch.
                batch_start = range_start * block_size
                batch_end = min(range_end * block_size, n_gauss_in_tile)
                if batch_start >= n_gauss_in_tile:
                    continue

                for k in range(batch_start, batch_end):
                    isect_idx = tile_start + k
                    fid = flatten_ids[isect_idx].item()
                    gauss_id = fid % N

                    # Get Gaussian params
                    mu_x = means2d_flat[fid, 0].item()
                    mu_y = means2d_flat[fid, 1].item()
                    c0 = conics_flat[fid, 0].item()
                    c1 = conics_flat[fid, 1].item()
                    c2 = conics_flat[fid, 2].item()
                    opacity = opacities_flat[fid].item()

                    # Check each pixel in the tile
                    for py_off in range(tile_size):
                        for px_off in range(tile_size):
                            px = tx * tile_size + px_off
                            py = ty * tile_size + py_off
                            if px >= image_width or py >= image_height:
                                continue

                            # Transmittance check
                            T = trans_flat[img_id, py, px].item()
                            if T < TRANSMITTANCE_THRESHOLD:
                                continue

                            # Evaluate Gaussian
                            dx = px + 0.5 - mu_x
                            dy = py + 0.5 - mu_y
                            sigma = (
                                0.5 * (c0 * dx * dx + c2 * dy * dy)
                                + c1 * dx * dy
                            )
                            if sigma < 0.0:
                                continue
                            alpha = min(0.999, opacity * math.exp(-sigma))
                            if alpha < ALPHA_THRESHOLD:
                                continue

                            pixel_id = py * image_width + px
                            out_gauss_ids_list.append(gauss_id)
                            out_pixel_ids_list.append(pixel_id)
                            out_image_ids_list.append(img_id)

    if len(out_gauss_ids_list) == 0:
        return (
            torch.zeros(0, dtype=torch.int64, device=device),
            torch.zeros(0, dtype=torch.int64, device=device),
            torch.zeros(0, dtype=torch.int64, device=device),
        )

    return (
        torch.tensor(out_gauss_ids_list, dtype=torch.int64, device=device),
        torch.tensor(out_pixel_ids_list, dtype=torch.int64, device=device),
        torch.tensor(out_image_ids_list, dtype=torch.int64, device=device),
    )


def fully_fused_projection_with_ut(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    opacities: Optional[Tensor],  # [..., N]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    calc_compensations: bool = False,
    camera_model: CameraModel = "pinhole",
    ut_params: Optional[UnscentedTransformParameters] = None,
    # distortion
    radial_coeffs: Optional[Tensor] = None,  # [..., C, 6] or [..., C, 4]
    tangential_coeffs: Optional[Tensor] = None,  # [..., C, 2]
    thin_prism_coeffs: Optional[Tensor] = None,  # [..., C, 4]
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
    lidar_coeffs: Optional[RowOffsetStructuredSpinningLidarModelParametersExt] = None,
    external_distortion_coeffs: Optional[BivariateWindshieldModelParameters] = None,
    # rolling shutter
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: Optional[Tensor] = None,  # [..., C, 4, 4]
    global_z_order: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Projects Gaussians to 2D using Unscented Transform (UT).

    Delegates to ``_torch_impl_ut._fully_fused_projection_with_ut``.

    .. todo:: MPS: Replace with a Metal kernel for better performance.
        CUDA equivalent: ``ProjectionUT3DGSFused.cu``

    Args:
        global_z_order: If True, sort by z-coordinate; if False, by Euclidean
            distance. Default: True.

    Returns:
        A tuple:

        - **radii**. [..., C, N, 2]
        - **means2d**. [..., C, N, 2]
        - **depths**. [..., C, N]
        - **conics**. [..., C, N, 3]
        - **compensations**. [..., C, N] or None
    """
    from gsplat.cuda._torch_impl_ut import _fully_fused_projection_with_ut

    if ut_params is None:
        ut_params = UnscentedTransformParameters()

    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    assert means.shape == batch_dims + (N, 3), means.shape
    assert quats.shape == batch_dims + (N, 4), quats.shape
    assert scales.shape == batch_dims + (N, 3), scales.shape
    if opacities is not None:
        assert opacities.shape == batch_dims + (N,), opacities.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    if radial_coeffs is not None:
        assert radial_coeffs.shape[:-1] == batch_dims + (C,) and radial_coeffs.shape[
            -1
        ] in [6, 4], radial_coeffs.shape
    if tangential_coeffs is not None:
        assert tangential_coeffs.shape == batch_dims + (C, 2), tangential_coeffs.shape
    if thin_prism_coeffs is not None:
        assert thin_prism_coeffs.shape == batch_dims + (C, 4), thin_prism_coeffs.shape
    if viewmats_rs is not None:
        assert viewmats_rs.shape == batch_dims + (C, 4, 4), viewmats_rs.shape

    if lidar_coeffs is not None:
        assert isinstance(
            lidar_coeffs, RowOffsetStructuredSpinningLidarModelParametersExt
        )

    return _fully_fused_projection_with_ut(
        means=means.contiguous(),
        quats=quats.contiguous(),
        scales=scales.contiguous(),
        opacities=opacities.contiguous() if opacities is not None else None,
        viewmats=viewmats.contiguous(),
        Ks=Ks.contiguous(),
        width=width,
        height=height,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        calc_compensations=calc_compensations,
        camera_model=camera_model,
        ut_params=ut_params,
        radial_coeffs=radial_coeffs.contiguous() if radial_coeffs is not None else None,
        tangential_coeffs=tangential_coeffs.contiguous() if tangential_coeffs is not None else None,
        thin_prism_coeffs=thin_prism_coeffs.contiguous() if thin_prism_coeffs is not None else None,
        ftheta_coeffs=ftheta_coeffs,
        lidar_coeffs=lidar_coeffs,
        rolling_shutter=rolling_shutter,
        viewmats_rs=viewmats_rs.contiguous() if viewmats_rs is not None else None,
        global_z_order=global_z_order,
    )


# ======================================================================
# 2DGS
# ======================================================================


def fully_fused_projection_2dgs(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
) -> Tuple[Tensor, ...]:
    """Projects 2D Gaussians to image space.

    Delegates to ``_torch_impl_2dgs._fully_fused_projection_2dgs``.

    .. todo:: MPS: Replace with fused Metal kernels for forward + backward.
        CUDA equivalents: ``Projection2DGSFused.cu`` (dense),
        ``Projection2DGSPacked.cu`` (packed).
    """
    from gsplat.cuda._torch_impl_2dgs import _fully_fused_projection_2dgs

    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    assert means.shape == batch_dims + (N, 3), means.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    means = means.contiguous()
    assert quats is not None, "quats is required"
    assert scales is not None, "scales is required"
    assert quats.shape == batch_dims + (N, 4), quats.shape
    assert scales.shape == batch_dims + (N, 3), scales.shape
    quats = quats.contiguous()
    scales = scales.contiguous()
    if sparse_grad:
        assert packed, "sparse_grad is only supported when packed is True"

    viewmats = viewmats.contiguous()
    Ks = Ks.contiguous()

    radii, means2d, depths, ray_transforms, normals = _fully_fused_projection_2dgs(
        means, quats, scales, viewmats, Ks, width, height,
        near_plane=near_plane,
        far_plane=far_plane,
        eps=eps2d,
    )

    if not packed:
        return radii, means2d, depths, ray_transforms, normals

    # --- packed mode ---
    valid = (radii > 0).any(dim=-1)  # [..., C, N]
    B = math.prod(batch_dims)
    radii_flat = radii.reshape(B, C, N, 2)
    means2d_flat = means2d.reshape(B, C, N, 2)
    depths_flat = depths.reshape(B, C, N)
    ray_transforms_flat = ray_transforms.reshape(B, C, N, 3, 3)
    normals_flat = normals.reshape(B, C, N, 3)
    valid_flat = valid.reshape(B, C, N)

    batch_idx, camera_idx, gauss_idx = torch.where(valid_flat)
    batch_ids = batch_idx.int()
    camera_ids = camera_idx.int()
    gaussian_ids = gauss_idx.int()

    packed_radii = radii_flat[batch_idx, camera_idx, gauss_idx]
    packed_means2d = means2d_flat[batch_idx, camera_idx, gauss_idx]
    packed_depths = depths_flat[batch_idx, camera_idx, gauss_idx]
    packed_ray_transforms = ray_transforms_flat[batch_idx, camera_idx, gauss_idx]
    packed_normals = normals_flat[batch_idx, camera_idx, gauss_idx]

    return (
        batch_ids,
        camera_ids,
        gaussian_ids,
        packed_radii,
        packed_means2d,
        packed_depths,
        packed_ray_transforms,
        packed_normals,
    )


def rasterize_to_pixels_2dgs(
    means2d: Tensor,  # [..., N, 2]
    ray_transforms: Tensor,  # [..., N, 3, 3]
    colors: Tensor,  # [..., N, channels]
    opacities: Tensor,  # [..., N]
    normals: Tensor,  # [..., N, 3]
    densify: Tensor,  # [..., N, 2]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [..., channels]
    masks: Optional[Tensor] = None,  # [..., tile_height, tile_width]
    packed: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Rasterize 2D Gaussians to pixels.

    Delegates to ``_torch_impl_2dgs._rasterize_to_pixels_2dgs`` (the
    pure-PyTorch reference).

    .. todo:: MPS: Replace with fused Metal kernels for forward + backward.
        Also implement distortion loss and median depth computation.
        CUDA equivalents: ``RasterizeToPixels2DGSFwd.cu``,
        ``RasterizeToPixels2DGSBwd.cu``

    Returns:
        A tuple:

        - **Rendered colors**.      [..., image_height, image_width, channels]
        - **Rendered alphas**.      [..., image_height, image_width, 1]
        - **Rendered normals**.     [..., image_height, image_width, 3]
        - **Rendered distortion**.  [..., image_height, image_width, 1]
        - **Rendered median depth**.[..., image_height, image_width, 1]
    """
    from gsplat.cuda._torch_impl_2dgs import _rasterize_to_pixels_2dgs

    image_dims = means2d.shape[:-2]
    channels = colors.shape[-1]
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert ray_transforms.shape == (nnz, 3, 3), ray_transforms.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(-2)
        assert means2d.shape == image_dims + (N, 2), means2d.shape
        assert ray_transforms.shape == image_dims + (N, 3, 3), ray_transforms.shape
        assert colors.shape[:-2] == image_dims, colors.shape
        assert opacities.shape == image_dims + (N,), opacities.shape
    if backgrounds is not None:
        assert backgrounds.shape == image_dims + (channels,), backgrounds.shape
        backgrounds = backgrounds.contiguous()

    # Channel padding (mirrors CUDA wrapper)
    if channels > 512 or channels == 0:
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [
                colors[..., :-1],
                torch.empty(*colors.shape[:-1], padded_channels, device=device),
                colors[..., -1:],
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0

    tile_height, tile_width = isect_offsets.shape[-2:]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    # The pure-PyTorch reference returns only (colors, alphas, normals).
    # Distortion and median depth are not computed.
    (
        render_colors,
        render_alphas,
        render_normals,
    ) = _rasterize_to_pixels_2dgs(
        means2d=means2d.contiguous(),
        ray_transforms=ray_transforms.contiguous(),
        colors=colors.contiguous(),
        normals=normals.contiguous(),
        opacities=opacities.contiguous(),
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        isect_offsets=isect_offsets.contiguous(),
        flatten_ids=flatten_ids.contiguous(),
        backgrounds=backgrounds,
    )

    # Produce zero-filled tensors for distortion and median depth to match
    # the CUDA API return shape.
    render_distort = torch.zeros(
        *render_alphas.shape, device=device, dtype=render_alphas.dtype
    )
    render_median = torch.zeros(
        *render_alphas.shape, device=device, dtype=render_alphas.dtype
    )

    if padded_channels > 0:
        render_colors = torch.cat(
            [render_colors[..., : -padded_channels - 1], render_colors[..., -1:]],
            dim=-1,
        )

    return render_colors, render_alphas, render_normals, render_distort, render_median


@torch.no_grad()
def rasterize_to_indices_in_range_2dgs(
    range_start: int,
    range_end: int,
    transmittances: Tensor,  # [..., image_height, image_width]
    means2d: Tensor,  # [..., N, 2]
    ray_transforms: Tensor,  # [..., N, 3, 3]
    opacities: Tensor,  # [..., N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Rasterizes a batch of 2D Gaussians to images but only returns indices.

    Not yet implemented for MPS — raises ``NotImplementedError``.

    .. todo:: MPS: Implement as a Metal kernel.
        CUDA equivalent: ``RasterizeToIndices2DGS.cu``
    """
    # TODO: MPS: Implement as a Metal kernel (CUDA: RasterizeToIndices2DGS.cu)
    raise NotImplementedError(
        "rasterize_to_indices_in_range_2dgs is not yet supported on the MPS backend. "
        "It requires a native Metal kernel."
    )
