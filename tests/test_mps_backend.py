# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import pkgutil
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest
import torch

from gsplat.mps import build as mps_build


def _reload_mps_backend():
    module = importlib.import_module("gsplat.mps._backend")
    return importlib.reload(module)


def _lookup_tolerance(tol: Any, key: Any, index: int) -> Any:
    if tol is None or isinstance(tol, (int, float)):
        return tol
    if isinstance(tol, Mapping):
        if key in tol:
            return tol[key]
        if index in tol:
            return tol[index]
        return None
    if isinstance(tol, Sequence) and not isinstance(tol, (str, bytes)):
        if index < len(tol):
            return tol[index]
        return None
    raise TypeError(
        f"rtol/atol must be None, a scalar, a mapping, or a sequence; got {type(tol).__name__}"
    )


def _compare_pair(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: Any,
    atol: Any,
    equal_nan: bool,
    check_dtype: bool,
) -> None:
    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise TypeError(
            f"[{name}] assert_mps_matches_reference compares tensors; "
            f"got {type(actual).__name__} vs {type(expected).__name__}"
        )
    actual_cpu = actual.detach().to("cpu")
    expected_cpu = expected.detach().to("cpu")
    try:
        torch.testing.assert_close(
            actual_cpu,
            expected_cpu,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            check_device=False,
            check_dtype=check_dtype,
        )
    except AssertionError as exc:
        raise AssertionError(f"[{name}] {exc}") from None


def assert_mps_matches_reference(
    mps_out,
    ref_out,
    *,
    rtol=None,
    atol=None,
    names=None,
    equal_nan: bool = False,
    check_dtype: bool = True,
) -> None:
    """Forward parity assertion for MPS kernel outputs vs a reference.

    Accepts a single tensor pair, a sequence of tensor pairs, or a dict of
    tensor pairs. ``rtol``/``atol`` may each be ``None``, a scalar applied
    uniformly, a mapping keyed by tensor name / index, or a positional
    sequence (when the inputs are sequences). Missing per-tensor entries fall
    back to ``torch.testing.assert_close`` defaults.

    MPS tensors are moved to CPU before comparison so this helper can be
    called with a reference tensor produced on CPU or CUDA.
    """

    if isinstance(mps_out, torch.Tensor) and isinstance(ref_out, torch.Tensor):
        _compare_pair(
            names if isinstance(names, str) else "tensor",
            mps_out,
            ref_out,
            _lookup_tolerance(rtol, "tensor", 0),
            _lookup_tolerance(atol, "tensor", 0),
            equal_nan,
            check_dtype,
        )
        return

    if isinstance(mps_out, Mapping) and isinstance(ref_out, Mapping):
        mps_keys = set(mps_out.keys())
        ref_keys = set(ref_out.keys())
        if mps_keys != ref_keys:
            raise TypeError(
                f"assert_mps_matches_reference: dict keys differ "
                f"(mps_out={sorted(mps_keys)!r}, ref_out={sorted(ref_keys)!r})"
            )
        for index, key in enumerate(mps_out):
            _compare_pair(
                str(key),
                mps_out[key],
                ref_out[key],
                _lookup_tolerance(rtol, key, index),
                _lookup_tolerance(atol, key, index),
                equal_nan,
                check_dtype,
            )
        return

    if (
        isinstance(mps_out, Sequence)
        and isinstance(ref_out, Sequence)
        and not isinstance(mps_out, (str, bytes))
        and not isinstance(ref_out, (str, bytes))
    ):
        if len(mps_out) != len(ref_out):
            raise TypeError(
                f"assert_mps_matches_reference: sequence lengths differ "
                f"(mps_out={len(mps_out)}, ref_out={len(ref_out)})"
            )
        resolved_names = names if names is not None else [None] * len(mps_out)
        if len(resolved_names) != len(mps_out):
            raise TypeError(
                f"assert_mps_matches_reference: names length {len(resolved_names)} "
                f"does not match sequence length {len(mps_out)}"
            )
        for index, (actual, expected) in enumerate(zip(mps_out, ref_out)):
            label = resolved_names[index] if resolved_names[index] is not None else f"[{index}]"
            _compare_pair(
                str(label),
                actual,
                expected,
                _lookup_tolerance(rtol, label, index),
                _lookup_tolerance(atol, label, index),
                equal_nan,
                check_dtype,
            )
        return

    raise TypeError(
        "assert_mps_matches_reference expects both arguments to be the same kind "
        "(tensor / sequence / mapping); got "
        f"mps_out={type(mps_out).__name__}, ref_out={type(ref_out).__name__}"
    )


def _as_tensor_tuple(out) -> tuple:
    if isinstance(out, torch.Tensor):
        return (out,)
    if isinstance(out, Sequence) and not isinstance(out, (str, bytes)):
        return tuple(out)
    raise TypeError(
        "assert_mps_grads_match_reference expects callables to return a tensor "
        f"or a sequence of tensors; got {type(out).__name__}"
    )


def _grad_input_indices(inputs: Sequence[torch.Tensor]) -> tuple:
    return tuple(
        i for i, t in enumerate(inputs)
        if isinstance(t, torch.Tensor) and t.requires_grad
    )


def _build_grad_outputs(outputs: tuple, seed: int = 0) -> list:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vectors = []
    for out in outputs:
        cpu_vec = torch.randn(
            out.shape, generator=generator, dtype=out.dtype, device="cpu"
        )
        vectors.append(cpu_vec.to(out.device))
    return vectors


def assert_mps_grads_match_reference(
    mps_fn,
    ref_fn,
    mps_inputs,
    ref_inputs=None,
    *,
    grad_outputs=None,
    rtol=None,
    atol=None,
    names=None,
    equal_nan: bool = False,
    check_dtype: bool = True,
    retain_graph: bool = False,
    allow_unused: bool = False,
) -> None:
    """Backward parity assertion for MPS kernels vs a reference implementation.

    Runs ``torch.autograd.grad`` on both ``mps_fn(*mps_inputs)`` and
    ``ref_fn(*ref_inputs)`` and compares the resulting gradient tuples using
    :func:`assert_mps_matches_reference`, so the ``rtol`` / ``atol`` / ``names``
    conventions match the forward helper.

    ``ref_inputs`` defaults to ``mps_inputs``; pass a parallel tuple when the
    reference runs on a different device. ``grad_outputs``, when omitted, is
    generated deterministically from a seed-0 CPU generator so the comparison
    is reproducible across devices.
    """

    mps_inputs = tuple(mps_inputs)
    ref_inputs = tuple(mps_inputs if ref_inputs is None else ref_inputs)

    if len(mps_inputs) != len(ref_inputs):
        raise TypeError(
            "assert_mps_grads_match_reference: mps_inputs and ref_inputs "
            f"have different lengths ({len(mps_inputs)} vs {len(ref_inputs)})"
        )

    mps_grad_idx = _grad_input_indices(mps_inputs)
    ref_grad_idx = _grad_input_indices(ref_inputs)
    if mps_grad_idx != ref_grad_idx:
        raise TypeError(
            "assert_mps_grads_match_reference: requires_grad pattern differs "
            f"(mps={mps_grad_idx}, ref={ref_grad_idx})"
        )
    if not mps_grad_idx:
        raise TypeError(
            "assert_mps_grads_match_reference: no input has requires_grad=True"
        )

    mps_out = _as_tensor_tuple(mps_fn(*mps_inputs))
    ref_out = _as_tensor_tuple(ref_fn(*ref_inputs))
    if len(mps_out) != len(ref_out):
        raise TypeError(
            "assert_mps_grads_match_reference: output arity differs "
            f"(mps={len(mps_out)}, ref={len(ref_out)})"
        )

    if grad_outputs is None:
        mps_grad_outputs = _build_grad_outputs(mps_out)
        ref_grad_outputs = _build_grad_outputs(ref_out)
    else:
        if len(grad_outputs) != len(mps_out):
            raise TypeError(
                "assert_mps_grads_match_reference: grad_outputs length "
                f"{len(grad_outputs)} does not match output count {len(mps_out)}"
            )
        mps_grad_outputs = [g.to(o.device) for g, o in zip(grad_outputs, mps_out)]
        ref_grad_outputs = [g.to(o.device) for g, o in zip(grad_outputs, ref_out)]

    mps_grad_inputs = tuple(mps_inputs[i] for i in mps_grad_idx)
    ref_grad_inputs = tuple(ref_inputs[i] for i in ref_grad_idx)

    mps_grads = torch.autograd.grad(
        outputs=mps_out,
        inputs=mps_grad_inputs,
        grad_outputs=mps_grad_outputs,
        retain_graph=retain_graph,
        allow_unused=allow_unused,
    )
    ref_grads = torch.autograd.grad(
        outputs=ref_out,
        inputs=ref_grad_inputs,
        grad_outputs=ref_grad_outputs,
        retain_graph=retain_graph,
        allow_unused=allow_unused,
    )

    resolved_names = (
        list(names) if names is not None
        else [f"grad[{i}]" for i in mps_grad_idx]
    )

    assert_mps_matches_reference(
        tuple(mps_grads),
        tuple(ref_grads),
        rtol=rtol,
        atol=atol,
        names=resolved_names,
        equal_nan=equal_nan,
        check_dtype=check_dtype,
    )


def test_load_metal_source_bundle_is_deterministic(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()

    (tmp_path / "b.metal").write_text(
        "kernel void second(device float* out [[buffer(0)]], uint idx "
        "[[thread_position_in_grid]]) { out[idx] = 1.0f; }\n",
        encoding="utf-8",
    )
    (nested / "a.metal").write_text(
        "kernel void first(device float* out [[buffer(0)]], uint idx "
        "[[thread_position_in_grid]]) { out[idx] = 0.0f; }\n",
        encoding="utf-8",
    )

    bundle = mps_build.load_metal_source_bundle(tmp_path)
    assert bundle.source_paths == ("b.metal", "nested/a.metal")
    assert bundle.kernel_names == ("second", "first")

    same_bundle = mps_build.load_metal_source_bundle(tmp_path)
    assert same_bundle.source_hash == bundle.source_hash

    (nested / "a.metal").write_text(
        "kernel void first(device float* out [[buffer(0)]], uint idx "
        "[[thread_position_in_grid]]) { out[idx] = 2.0f; }\n",
        encoding="utf-8",
    )
    changed_bundle = mps_build.load_metal_source_bundle(tmp_path)
    assert changed_bundle.source_hash != bundle.source_hash


def test_build_and_load_gsplat_uses_source_hash_cache(tmp_path, monkeypatch):
    (tmp_path / "cache_test.metal").write_text(
        "kernel void cache_test(device float* out [[buffer(0)]], uint idx "
        "[[thread_position_in_grid]]) { out[idx] = 0.0f; }\n",
        encoding="utf-8",
    )

    calls = []

    def fake_compile_shader(source):
        calls.append(source)
        return SimpleNamespace(cache_test=lambda *args, **kwargs: None)

    mps_build.clear_mps_backend_cache()
    monkeypatch.setattr(
        torch, "mps", SimpleNamespace(compile_shader=fake_compile_shader), raising=False
    )

    first = mps_build.build_and_load_gsplat(tmp_path)
    second = mps_build.build_and_load_gsplat(tmp_path)

    assert first is second
    assert len(calls) == 1
    assert first.has_kernel("cache_test")


def test_bootstrap_shader_is_packaged():
    data = pkgutil.get_data("gsplat.mps", "csrc/bootstrap.metal")
    assert data is not None
    assert b"gsplat_bootstrap_fill_float" in data


def test_backend_keeps_cpu_only_imports_working(monkeypatch):
    mps_build.clear_mps_backend_cache()
    monkeypatch.setattr(
        torch.backends, "mps", SimpleNamespace(is_available=lambda: False), raising=False
    )

    backend = _reload_mps_backend()
    assert backend._C is None


@pytest.mark.parametrize("mode", ["missing_compile_shader", "compile_failure"])
def test_backend_init_errors_are_explicit(monkeypatch, mode):
    mps_build.clear_mps_backend_cache()
    monkeypatch.setattr(
        torch.backends, "mps", SimpleNamespace(is_available=lambda: True), raising=False
    )

    if mode == "missing_compile_shader":
        monkeypatch.setattr(torch, "mps", SimpleNamespace(), raising=False)
        match = r"torch\.mps\.compile_shader"
    else:
        monkeypatch.setattr(
            torch,
            "mps",
            SimpleNamespace(
                compile_shader=lambda source: (_ for _ in ()).throw(
                    RuntimeError("synthetic compile failure")
                )
            ),
            raising=False,
        )
        match = r"synthetic compile failure"

    with pytest.raises(RuntimeError, match=match):
        _reload_mps_backend()


@pytest.mark.skipif(
    not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "compile_shader")
    ),
    reason="Native MPS shader runtime is unavailable",
)
def test_backend_initializes_bootstrap_shader_on_mps():
    mps_build.clear_mps_backend_cache()
    backend = _reload_mps_backend()

    assert backend._C is not None
    assert mps_build.BOOTSTRAP_SHADER_NAME in backend._C.kernel_names
    assert backend._C.get_kernel(mps_build.BOOTSTRAP_SHADER_NAME) is not None


def test_assert_mps_matches_reference_single_tensor_scalar_tolerance():
    actual = torch.tensor([1.0, 2.0, 3.0])
    expected = actual + 1e-6
    assert_mps_matches_reference(actual, expected, rtol=1e-4, atol=1e-4)

    with pytest.raises(AssertionError, match=r"\[tensor\]"):
        assert_mps_matches_reference(actual, actual + 1.0, rtol=1e-6, atol=1e-6)


def test_assert_mps_matches_reference_sequence_per_index_tolerance():
    loose = torch.zeros(4)
    loose_ref = loose + 5e-3
    tight = torch.ones(4)
    tight_ref = tight + 1e-6

    assert_mps_matches_reference(
        (loose, tight),
        (loose_ref, tight_ref),
        rtol={0: 1e-2, 1: 1e-5},
        atol={0: 1e-2, 1: 1e-5},
        names=["loose", "tight"],
    )

    with pytest.raises(AssertionError, match=r"\[tight\]"):
        assert_mps_matches_reference(
            (loose, tight),
            (loose_ref, tight + 1.0),
            rtol={0: 1e-2, 1: 1e-5},
            atol={0: 1e-2, 1: 1e-5},
            names=["loose", "tight"],
        )


def test_assert_mps_matches_reference_dict_per_key_tolerance():
    good = {
        "means2d": torch.zeros(3, 2),
        "conics": torch.ones(3, 3),
    }
    perturbed = {
        "means2d": torch.zeros(3, 2),
        "conics": torch.ones(3, 3) + 1.0,
    }

    assert_mps_matches_reference(
        good,
        {"means2d": good["means2d"] + 1e-6, "conics": good["conics"] + 1e-6},
        rtol={"means2d": 1e-4, "conics": 1e-4},
        atol={"means2d": 1e-4, "conics": 1e-4},
    )

    with pytest.raises(AssertionError, match=r"\[conics\]"):
        assert_mps_matches_reference(good, perturbed, rtol=1e-6, atol=1e-6)

    with pytest.raises(TypeError, match="dict keys differ"):
        assert_mps_matches_reference(good, {"means2d": good["means2d"]})


def test_assert_mps_grads_match_reference_single_input_scalar_tolerance():
    def good_fn(x):
        return x * x

    def bad_fn(x):
        return x * x + x

    x_mps = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    x_ref = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    assert_mps_grads_match_reference(
        good_fn, good_fn, (x_mps,), (x_ref,), rtol=1e-6, atol=1e-6
    )

    x_mps2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    x_ref2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    with pytest.raises(AssertionError, match=r"\[grad\[0\]\]"):
        assert_mps_grads_match_reference(
            good_fn, bad_fn, (x_mps2,), (x_ref2,), rtol=1e-6, atol=1e-6
        )


def test_assert_mps_grads_match_reference_per_index_tolerance():
    def good_fn(a, b):
        return a * 2.0 + b * b

    def bad_fn(a, b):
        return a * 2.0 + b * b + b

    a_mps = torch.tensor([1.0, 2.0], requires_grad=True)
    b_mps = torch.tensor([3.0, 4.0], requires_grad=True)
    a_ref = torch.tensor([1.0, 2.0], requires_grad=True)
    b_ref = torch.tensor([3.0, 4.0], requires_grad=True)

    assert_mps_grads_match_reference(
        good_fn, good_fn,
        (a_mps, b_mps), (a_ref, b_ref),
        rtol={0: 1e-3, 1: 1e-6},
        atol={0: 1e-3, 1: 1e-6},
        names=["coarse", "fine"],
    )

    a_mps2 = torch.tensor([1.0, 2.0], requires_grad=True)
    b_mps2 = torch.tensor([3.0, 4.0], requires_grad=True)
    a_ref2 = torch.tensor([1.0, 2.0], requires_grad=True)
    b_ref2 = torch.tensor([3.0, 4.0], requires_grad=True)
    with pytest.raises(AssertionError, match=r"\[fine\]"):
        assert_mps_grads_match_reference(
            good_fn, bad_fn,
            (a_mps2, b_mps2), (a_ref2, b_ref2),
            rtol={0: 1e-3, 1: 1e-6},
            atol={0: 1e-3, 1: 1e-6},
            names=["coarse", "fine"],
        )


def test_assert_mps_grads_match_reference_deterministic_without_grad_outputs():
    def fn(x):
        return (x * x, x.sin())

    for _ in range(2):
        x_mps = torch.tensor([0.5, 1.5, 2.5], requires_grad=True)
        x_ref = torch.tensor([0.5, 1.5, 2.5], requires_grad=True)
        assert_mps_grads_match_reference(
            fn, fn, (x_mps,), (x_ref,), rtol=1e-6, atol=1e-6
        )


# ---------------------------------------------------------------------------
# T1.1 — quat_scale_to_covar_preci native MPS kernel parity
# ---------------------------------------------------------------------------


_SKIP_NO_MPS = pytest.mark.skipif(
    not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "compile_shader")
    ),
    reason="Native MPS shader runtime is unavailable",
)


def _make_quat_scale_inputs(N: int, device: torch.device, seed: int = 0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    quats_cpu = torch.randn(N, 4, generator=gen)
    scales_cpu = torch.rand(N, 3, generator=gen) * 2.0 + 0.1  # avoid tiny scales
    return quats_cpu.to(device), scales_cpu.to(device)


@_SKIP_NO_MPS
@pytest.mark.parametrize("N", [1, 1024, 100_000])
@pytest.mark.parametrize("triu", [False, True])
@pytest.mark.parametrize(
    "compute_covar,compute_preci",
    [(True, True), (True, False), (False, True)],
)
def test_quat_scale_to_covar_preci_forward_parity(N, triu, compute_covar, compute_preci):
    from gsplat.cuda._math import _quat_scale_to_covar_preci
    from gsplat.mps._wrapper import quat_scale_to_covar_preci

    device = torch.device("mps")
    quats, scales = _make_quat_scale_inputs(N, device)

    covars, precis = quat_scale_to_covar_preci(
        quats, scales, compute_covar, compute_preci, triu
    )
    c_ref, p_ref = _quat_scale_to_covar_preci(
        quats, scales, compute_covar, compute_preci, triu
    )

    if compute_covar:
        assert_mps_matches_reference(covars, c_ref, rtol=1e-4, atol=1e-5, names="covars")
    else:
        assert covars is None
    if compute_preci:
        assert_mps_matches_reference(precis, p_ref, rtol=1e-3, atol=1e-4, names="precis")
    else:
        assert precis is None


@_SKIP_NO_MPS
@pytest.mark.parametrize("N", [1, 1024])
@pytest.mark.parametrize("triu", [False, True])
@pytest.mark.parametrize(
    "compute_covar,compute_preci",
    [(True, True), (True, False), (False, True)],
)
def test_quat_scale_to_covar_preci_backward_parity(N, triu, compute_covar, compute_preci):
    from gsplat.cuda._math import _quat_scale_to_covar_preci
    from gsplat.mps._wrapper import quat_scale_to_covar_preci

    device = torch.device("mps")
    quats, scales = _make_quat_scale_inputs(N, device, seed=1)

    def mps_fn(q, s):
        c, p = quat_scale_to_covar_preci(q, s, compute_covar, compute_preci, triu)
        # Filter out the None slots so autograd only sees real tensors.
        return tuple(t for t in (c, p) if t is not None)

    def ref_fn(q, s):
        c, p = _quat_scale_to_covar_preci(q, s, compute_covar, compute_preci, triu)
        return tuple(t for t in (c, p) if t is not None)

    q_mps = quats.detach().clone().requires_grad_(True)
    s_mps = scales.detach().clone().requires_grad_(True)
    q_ref = quats.detach().clone().requires_grad_(True)
    s_ref = scales.detach().clone().requires_grad_(True)

    assert_mps_grads_match_reference(
        mps_fn,
        ref_fn,
        (q_mps, s_mps),
        (q_ref, s_ref),
        rtol=1e-3,
        atol=1e-4,
        names=["v_quats", "v_scales"],
    )


@_SKIP_NO_MPS
def test_quat_scale_to_covar_preci_handles_non_normalized_quats():
    """The kernel must internally normalize via rsqrt, matching the oracle."""
    from gsplat.cuda._math import _quat_scale_to_covar_preci
    from gsplat.mps._wrapper import quat_scale_to_covar_preci

    device = torch.device("mps")
    quats, scales = _make_quat_scale_inputs(32, device, seed=7)
    # Blow up the magnitudes to exercise the rsqrt normalization path.
    quats = quats * 17.3

    covars, precis = quat_scale_to_covar_preci(quats, scales, True, True, False)
    c_ref, p_ref = _quat_scale_to_covar_preci(quats, scales, True, True, False)
    assert_mps_matches_reference(
        (covars, precis),
        (c_ref, p_ref),
        rtol=1e-4,
        atol=1e-5,
        names=["covars", "precis"],
    )


# ---------------------------------------------------------------------------
# T1.2 — spherical_harmonics
# ---------------------------------------------------------------------------


def _make_sh_inputs(N: int, K: int, device: torch.device, seed: int = 0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dirs_cpu = torch.randn(N, 3, generator=gen)
    coeffs_cpu = torch.randn(N, K, 3, generator=gen) * 0.3
    return dirs_cpu.to(device), coeffs_cpu.to(device)


@_SKIP_NO_MPS
@pytest.mark.parametrize("N", [1, 1024, 100_000])
@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("with_mask", [False, True])
def test_spherical_harmonics_forward_parity(N, degree, with_mask):
    from gsplat.cuda._torch_impl import _spherical_harmonics
    from gsplat.mps._wrapper import spherical_harmonics

    K = (degree + 1) ** 2
    device = torch.device("mps")
    dirs, coeffs = _make_sh_inputs(N, K, device, seed=2)

    if with_mask:
        gen = torch.Generator(device="cpu").manual_seed(N + degree + 7)
        masks_cpu = torch.rand(N, generator=gen) > 0.3  # ~70% kept
        masks = masks_cpu.to(device)
    else:
        masks = None

    mps_out = spherical_harmonics(degree, dirs, coeffs, masks=masks)
    # The pure-PyTorch oracle does not support masks — compute the full
    # reference then zero out masked rows so we can compare bit-for-bit.
    ref_full = _spherical_harmonics(degree, dirs, coeffs)
    if with_mask:
        ref_full = ref_full * masks.to(ref_full.dtype).unsqueeze(-1)

    assert_mps_matches_reference(
        mps_out, ref_full, rtol=1e-4, atol=1e-5, names="colors"
    )


@_SKIP_NO_MPS
@pytest.mark.parametrize("N", [1, 1024])
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_spherical_harmonics_backward_parity(N, degree):
    # degree=0 is omitted: band 0 is constant in dirs, so the autograd graph
    # has no dirs dependency and torch.autograd.grad raises. Coeff-only
    # backward at degree 0 is covered by test_spherical_harmonics_backward_skips_v_dirs_when_frozen
    # (via the same _SphericalHarmonics.backward code path).
    from gsplat.cuda._torch_impl import _spherical_harmonics
    from gsplat.mps._wrapper import spherical_harmonics

    K = (degree + 1) ** 2
    device = torch.device("mps")
    dirs, coeffs = _make_sh_inputs(N, K, device, seed=3)

    def mps_fn(d, c):
        return spherical_harmonics(degree, d, c)

    def ref_fn(d, c):
        return _spherical_harmonics(degree, d, c)

    d_mps = dirs.detach().clone().requires_grad_(True)
    c_mps = coeffs.detach().clone().requires_grad_(True)
    d_ref = dirs.detach().clone().requires_grad_(True)
    c_ref = coeffs.detach().clone().requires_grad_(True)

    assert_mps_grads_match_reference(
        mps_fn,
        ref_fn,
        (d_mps, c_mps),
        (d_ref, c_ref),
        rtol=1e-3,
        atol=1e-4,
        names=["v_dirs", "v_coeffs"],
    )


@_SKIP_NO_MPS
def test_spherical_harmonics_backward_skips_v_dirs_when_frozen():
    """dirs.requires_grad=False must return (None, v_coeffs) with only v_coeffs populated."""
    from gsplat.cuda._torch_impl import _spherical_harmonics
    from gsplat.mps._wrapper import spherical_harmonics

    degree = 3
    K = (degree + 1) ** 2
    N = 64
    device = torch.device("mps")
    dirs, coeffs = _make_sh_inputs(N, K, device, seed=4)

    c_mps = coeffs.detach().clone().requires_grad_(True)
    c_ref = coeffs.detach().clone().requires_grad_(True)

    out_mps = spherical_harmonics(degree, dirs, c_mps)
    out_ref = _spherical_harmonics(degree, dirs, c_ref)

    g_out = torch.randn_like(out_mps)
    (v_coeffs_mps,) = torch.autograd.grad(out_mps, (c_mps,), grad_outputs=g_out)
    (v_coeffs_ref,) = torch.autograd.grad(out_ref, (c_ref,), grad_outputs=g_out)

    assert_mps_matches_reference(
        v_coeffs_mps, v_coeffs_ref, rtol=1e-3, atol=1e-4, names="v_coeffs"
    )


@_SKIP_NO_MPS
def test_spherical_harmonics_forward_preserves_batch_dims():
    """Verify multi-dim batch shapes are flattened/reshaped correctly."""
    from gsplat.cuda._torch_impl import _spherical_harmonics
    from gsplat.mps._wrapper import spherical_harmonics

    degree = 2
    K = (degree + 1) ** 2
    device = torch.device("mps")
    gen = torch.Generator(device="cpu").manual_seed(5)
    dirs = torch.randn(4, 8, 3, generator=gen).to(device)
    coeffs = (torch.randn(4, 8, K, 3, generator=gen) * 0.3).to(device)

    mps_out = spherical_harmonics(degree, dirs, coeffs)
    ref_out = _spherical_harmonics(degree, dirs, coeffs)
    assert mps_out.shape == (4, 8, 3)
    assert_mps_matches_reference(mps_out, ref_out, rtol=1e-4, atol=1e-5)
