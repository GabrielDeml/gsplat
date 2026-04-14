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
