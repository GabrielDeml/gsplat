# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import pkgutil
from types import SimpleNamespace

import pytest
import torch

from gsplat.mps import build as mps_build


def _reload_mps_backend():
    module = importlib.import_module("gsplat.mps._backend")
    return importlib.reload(module)


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
