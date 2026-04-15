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

"""Runtime Metal shader loading for the gsplat MPS backend."""

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

PathLike = Union[str, os.PathLike]

METAL_SOURCE_SUBDIR = "csrc"
BOOTSTRAP_SHADER_NAME = "gsplat_bootstrap_fill_float"

_KERNEL_RE = re.compile(r"\bkernel\s+void\s+([A-Za-z_]\w*)\s*\(")
_MPS_BACKEND_CACHE: Dict[str, "MPSBackendHandle"] = {}


class MPSBuildError(RuntimeError):
    """Base class for gsplat MPS build/load failures."""


class MPSCompileShaderUnavailableError(MPSBuildError):
    """Raised when the local PyTorch build does not expose MPS shader JIT."""


class MPSShaderCompilationError(MPSBuildError):
    """Raised when packaged Metal sources fail to compile."""


@dataclass(frozen=True)
class MetalSourceBundle:
    """Resolved and concatenated Metal sources for one build."""

    root: Path
    source: str
    source_hash: str
    kernel_names: Tuple[str, ...]
    source_paths: Tuple[str, ...]


@dataclass(frozen=True)
class MPSBackendHandle:
    """Lightweight handle for a compiled gsplat MPS shader library."""

    library: Any
    source_hash: str
    kernel_names: Tuple[str, ...]
    source_paths: Tuple[str, ...]

    def has_kernel(self, name: str) -> bool:
        return name in self.kernel_names and hasattr(self.library, name)

    def get_kernel(self, name: str) -> Any:
        if name not in self.kernel_names:
            raise AttributeError(
                f"gsplat MPS kernel '{name}' is not part of the compiled shader library."
            )
        return getattr(self.library, name)


def clear_mps_backend_cache() -> None:
    """Clear the in-process backend cache.

    Intended for tests and interactive development only.
    """

    _MPS_BACKEND_CACHE.clear()


def get_metal_source_root(root: Optional[PathLike] = None) -> Path:
    """Return the directory containing packaged Metal sources."""

    source_root = (
        Path(root)
        if root is not None
        else Path(__file__).resolve().parent / METAL_SOURCE_SUBDIR
    )
    if not source_root.exists():
        raise FileNotFoundError(
            f"gsplat MPS source directory does not exist: {source_root}"
        )
    if not source_root.is_dir():
        raise NotADirectoryError(
            f"gsplat MPS source root must be a directory: {source_root}"
        )
    return source_root


def discover_metal_source_files(root: Optional[PathLike] = None) -> Tuple[Path, ...]:
    """Discover packaged ``.metal`` sources in deterministic order."""

    source_root = get_metal_source_root(root)
    paths = tuple(
        sorted(
            (path for path in source_root.rglob("*.metal") if path.is_file()),
            key=lambda path: path.relative_to(source_root).as_posix(),
        )
    )
    if not paths:
        raise FileNotFoundError(
            f"gsplat MPS backend found no Metal sources under {source_root}"
        )
    return paths


def load_metal_source_bundle(root: Optional[PathLike] = None) -> MetalSourceBundle:
    """Read and concatenate packaged Metal sources into one compile unit."""

    source_root = get_metal_source_root(root)
    paths = discover_metal_source_files(source_root)

    chunks = []
    source_paths = []
    for path in paths:
        relative_path = path.relative_to(source_root).as_posix()
        source_paths.append(relative_path)
        chunks.append(f"// BEGIN {relative_path}\n")
        chunks.append(path.read_text(encoding="utf-8").rstrip())
        chunks.append(f"\n// END {relative_path}\n")

    source = "".join(chunks)
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    kernel_names = tuple(dict.fromkeys(_KERNEL_RE.findall(source)))

    if not kernel_names:
        raise ValueError(
            f"gsplat MPS backend found no Metal kernels under {source_root}"
        )

    return MetalSourceBundle(
        root=source_root,
        source=source,
        source_hash=source_hash,
        kernel_names=kernel_names,
        source_paths=tuple(source_paths),
    )


def _compile_shader_library(source: str) -> Any:
    torch_mps = getattr(torch, "mps", None)
    compile_shader = getattr(torch_mps, "compile_shader", None)
    if compile_shader is None:
        raise MPSCompileShaderUnavailableError(
            "gsplat MPS native shader compilation requires a PyTorch build that "
            "exposes torch.mps.compile_shader(source)."
        )

    try:
        return compile_shader(source)
    except Exception as exc:
        raise MPSShaderCompilationError(
            "Failed to compile packaged gsplat Metal shaders via "
            "torch.mps.compile_shader(source)."
        ) from exc


def build_and_load_gsplat(root: Optional[PathLike] = None) -> MPSBackendHandle:
    """Compile packaged Metal sources and return a cached backend handle."""

    bundle = load_metal_source_bundle(root)

    cached_backend = _MPS_BACKEND_CACHE.get(bundle.source_hash)
    if cached_backend is not None:
        return cached_backend

    library = _compile_shader_library(bundle.source)
    backend = MPSBackendHandle(
        library=library,
        source_hash=bundle.source_hash,
        kernel_names=bundle.kernel_names,
        source_paths=bundle.source_paths,
    )
    missing_kernels = tuple(
        name for name in bundle.kernel_names if not backend.has_kernel(name)
    )
    if missing_kernels:
        raise MPSShaderCompilationError(
            "Compiled gsplat MPS shader library is missing expected kernels: "
            + ", ".join(missing_kernels)
        )
    _MPS_BACKEND_CACHE[bundle.source_hash] = backend
    return backend


__all__ = [
    "BOOTSTRAP_SHADER_NAME",
    "METAL_SOURCE_SUBDIR",
    "MPSBackendHandle",
    "MPSBuildError",
    "MPSCompileShaderUnavailableError",
    "MPSShaderCompilationError",
    "MetalSourceBundle",
    "build_and_load_gsplat",
    "clear_mps_backend_cache",
    "discover_metal_source_files",
    "get_metal_source_root",
    "load_metal_source_bundle",
]
