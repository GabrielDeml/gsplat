# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Forward-only benchmark harness for the MPS backend (T0.6).

Times each ``gsplat.mps._wrapper`` kernel on a fixed 100k-Gaussian scene and
compares MPS-device wall time vs CPU-device wall time. Every entry point
currently delegates to the pure-PyTorch oracle, so the initial numbers are a
pure-PyTorch MPS baseline. Once native Metal kernels land (T1.x), the MPS
column reflects the native kernel — no harness change needed.

Usage
-----
    python tests/bench_mps.py                       # defaults (N=100k, 512x512)
    python tests/bench_mps.py --n 10000 --reps 5    # smaller & faster
    python tests/bench_mps.py --kernels adam,sh     # subset
    python tests/bench_mps.py --write-docs          # also update docs/DEV.md

This is *not* a pytest test — it's slow and it's a timing tool. It's opt-in
from ``run_mps_tests.sh`` (see the trailing comment there).
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gsplat.mps import _wrapper as mps


TILE_SIZE = 16
SH_DEGREE = 3
SH_K = (SH_DEGREE + 1) ** 2  # 16
COLOR_CHANNELS = 3

# rasterize_to_indices_in_range is a pure-Python triple loop today; cap its
# scene so a single bench call stays under ~30s on CPU.
SLOW_KERNEL_N_CAP = 2_000
SLOW_KERNEL_WH_CAP = 128

DOCS_PATH = REPO_ROOT / "docs" / "DEV.md"
BENCH_START = "<!-- BENCH_MPS_START -->"
BENCH_END = "<!-- BENCH_MPS_END -->"


# ---------------------------------------------------------------------------
# Scene synthesis
# ---------------------------------------------------------------------------


@dataclass
class Scene:
    """Minimal 3DGS scene on a target device."""

    device: torch.device
    n: int
    width: int
    height: int
    means: torch.Tensor       # [N, 3]
    quats: torch.Tensor       # [N, 4] (normalized)
    scales: torch.Tensor      # [N, 3]
    opacities: torch.Tensor   # [N]
    colors: torch.Tensor      # [N, 3]
    sh_coeffs: torch.Tensor   # [N, SH_K, 3]
    sh_dirs: torch.Tensor     # [N, 3]
    viewmats: torch.Tensor    # [1, 4, 4]
    Ks: torch.Tensor          # [1, 3, 3]


def _norm(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return t / t.norm(dim=dim, keepdim=True).clamp_min(1e-12)


def build_scene(n: int, width: int, height: int, seed: int, device: torch.device) -> Scene:
    """Deterministic synthetic 3DGS scene — mirrors test_basic conventions."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    means = torch.randn(n, 3, generator=g) * 1.0
    quats = _norm(torch.randn(n, 4, generator=g))
    scales = torch.rand(n, 3, generator=g) * 0.09 + 0.01
    opacities = torch.rand(n, generator=g) * 0.9 + 0.05
    colors = torch.rand(n, 3, generator=g)
    sh_coeffs = torch.randn(n, SH_K, 3, generator=g) * 0.1
    sh_dirs = _norm(torch.randn(n, 3, generator=g))

    # Camera: looking at origin from z = -5; pinhole with fov ~60deg.
    viewmat = torch.eye(4)
    viewmat[2, 3] = 5.0  # camera at z=-5 in world ( translation of world→cam )
    viewmats = viewmat.unsqueeze(0)

    fx = fy = 0.5 * max(width, height) / math.tan(math.radians(30.0))
    K = torch.tensor([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])
    Ks = K.unsqueeze(0)

    return Scene(
        device=device,
        n=n,
        width=width,
        height=height,
        means=means.to(device),
        quats=quats.to(device),
        scales=scales.to(device),
        opacities=opacities.to(device),
        colors=colors.to(device),
        sh_coeffs=sh_coeffs.to(device),
        sh_dirs=sh_dirs.to(device),
        viewmats=viewmats.to(device),
        Ks=Ks.to(device),
    )


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def time_call(fn: Callable[[], object], device: torch.device, *, reps: int, warmup: int) -> float:
    """Returns median milliseconds over `reps` timed iterations."""
    for _ in range(warmup):
        fn()
    _sync(device)

    samples: List[float] = []
    for _ in range(reps):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


# ---------------------------------------------------------------------------
# Per-kernel benches
# ---------------------------------------------------------------------------


def bench_adam(scene: Scene, reps: int, warmup: int) -> float:
    device = scene.device
    param = scene.means.clone().contiguous()
    param_grad = torch.randn_like(param)
    exp_avg = torch.zeros_like(param)
    exp_avg_sq = torch.zeros_like(param)
    valid = torch.ones(scene.n, dtype=torch.bool, device=device)

    def step():
        # Reset buffers so repeated calls stay numerically bounded.
        exp_avg.zero_()
        exp_avg_sq.zero_()
        mps.adam(param, param_grad, exp_avg, exp_avg_sq, valid, 1e-3, 0.9, 0.999, 1e-8)

    return time_call(step, device, reps=reps, warmup=warmup)


def bench_spherical_harmonics(scene: Scene, reps: int, warmup: int) -> float:
    def call():
        mps.spherical_harmonics(SH_DEGREE, scene.sh_dirs, scene.sh_coeffs)

    return time_call(call, scene.device, reps=reps, warmup=warmup)


def bench_quat_scale_to_covar_preci(scene: Scene, reps: int, warmup: int) -> float:
    def call():
        mps.quat_scale_to_covar_preci(
            scene.quats, scene.scales, compute_covar=True, compute_preci=True, triu=False
        )

    return time_call(call, scene.device, reps=reps, warmup=warmup)


def bench_projection(scene: Scene, reps: int, warmup: int) -> float:
    def call():
        mps.fully_fused_projection(
            scene.means, None, scene.quats, scene.scales,
            scene.viewmats, scene.Ks, scene.width, scene.height,
        )

    return time_call(call, scene.device, reps=reps, warmup=warmup)


def _project(scene: Scene):
    """Run projection once and return outputs (shared by isect/rasterize)."""
    return mps.fully_fused_projection(
        scene.means, None, scene.quats, scene.scales,
        scene.viewmats, scene.Ks, scene.width, scene.height,
    )


def _tile_grid(scene: Scene) -> tuple[int, int]:
    tw = (scene.width + TILE_SIZE - 1) // TILE_SIZE
    th = (scene.height + TILE_SIZE - 1) // TILE_SIZE
    return tw, th


def bench_isect_tiles(scene: Scene, reps: int, warmup: int) -> float:
    radii, means2d, depths, _conics, _comp = _project(scene)
    tw, th = _tile_grid(scene)

    def call():
        mps.isect_tiles(means2d, radii, depths, TILE_SIZE, tw, th, sort=True)

    return time_call(call, scene.device, reps=reps, warmup=warmup)


def bench_isect_offset_encode(scene: Scene, reps: int, warmup: int) -> float:
    radii, means2d, depths, _conics, _comp = _project(scene)
    tw, th = _tile_grid(scene)
    _, isect_ids, _ = mps.isect_tiles(
        means2d, radii, depths, TILE_SIZE, tw, th, sort=True
    )
    n_images = scene.viewmats.shape[0]

    def call():
        mps.isect_offset_encode(isect_ids, n_images, tw, th)

    return time_call(call, scene.device, reps=reps, warmup=warmup)


def bench_rasterize_to_pixels(scene: Scene, reps: int, warmup: int) -> float:
    radii, means2d, depths, conics, _comp = _project(scene)
    tw, th = _tile_grid(scene)
    _, isect_ids, flatten_ids = mps.isect_tiles(
        means2d, radii, depths, TILE_SIZE, tw, th, sort=True
    )
    n_images = scene.viewmats.shape[0]
    isect_offsets = mps.isect_offset_encode(isect_ids, n_images, tw, th)

    # Broadcast per-Gaussian tensors to match [C, N, ...] that rasterize expects.
    C = scene.viewmats.shape[0]
    colors = scene.colors.unsqueeze(0).expand(C, -1, -1).contiguous()
    opacities = scene.opacities.unsqueeze(0).expand(C, -1).contiguous()

    def call():
        mps.rasterize_to_pixels(
            means2d, conics, colors, opacities,
            scene.width, scene.height, TILE_SIZE,
            isect_offsets, flatten_ids,
        )

    return time_call(call, scene.device, reps=reps, warmup=warmup)


def bench_rasterize_to_indices_in_range(scene: Scene, reps: int, warmup: int) -> float:
    radii, means2d, depths, conics, _comp = _project(scene)
    tw, th = _tile_grid(scene)
    _, isect_ids, flatten_ids = mps.isect_tiles(
        means2d, radii, depths, TILE_SIZE, tw, th, sort=True
    )
    n_images = scene.viewmats.shape[0]
    isect_offsets = mps.isect_offset_encode(isect_ids, n_images, tw, th)

    C = scene.viewmats.shape[0]
    opacities = scene.opacities.unsqueeze(0).expand(C, -1).contiguous()
    transmittances = torch.ones(C, scene.height, scene.width, device=scene.device)

    def call():
        mps.rasterize_to_indices_in_range(
            0, 1, transmittances, means2d, conics, opacities,
            scene.width, scene.height, TILE_SIZE, isect_offsets, flatten_ids,
        )

    return time_call(call, scene.device, reps=reps, warmup=warmup)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class KernelBench:
    key: str              # short CLI alias
    display: str          # display name in the table
    run: Callable[[Scene, int, int], float]
    slow: bool = False    # if True, cap scene size to keep runtime sane


KERNELS: list[KernelBench] = [
    KernelBench("adam", "adam", bench_adam),
    KernelBench("sh", "spherical_harmonics", bench_spherical_harmonics),
    KernelBench("quat_scale", "quat_scale_to_covar_preci", bench_quat_scale_to_covar_preci),
    KernelBench("proj", "fully_fused_projection", bench_projection),
    KernelBench("isect_tiles", "isect_tiles", bench_isect_tiles),
    KernelBench("isect_offset", "isect_offset_encode", bench_isect_offset_encode),
    KernelBench("rasterize", "rasterize_to_pixels", bench_rasterize_to_pixels),
    KernelBench(
        "rast_indices",
        "rasterize_to_indices_in_range",
        bench_rasterize_to_indices_in_range,
        slow=True,
    ),
]

KERNELS_BY_KEY = {k.key: k for k in KERNELS}


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


@dataclass
class Row:
    name: str
    n: int
    wh: str
    mps_ms: Optional[float]
    cpu_ms: Optional[float]

    @property
    def speedup(self) -> Optional[float]:
        if self.mps_ms is None or self.cpu_ms is None or self.mps_ms == 0.0:
            return None
        return self.cpu_ms / self.mps_ms


def _fmt_ms(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:,.2f}"


def _fmt_speedup(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.2f}×"


def render_table(rows: list[Row]) -> str:
    lines = [
        "| Kernel | N | W×H | MPS (ms) | CPU (ms) | Speedup |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.name}` | {r.n:,} | {r.wh} | {_fmt_ms(r.mps_ms)} | "
            f"{_fmt_ms(r.cpu_ms)} | {_fmt_speedup(r.speedup)} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# docs/DEV.md integration
# ---------------------------------------------------------------------------


def write_docs(table_md: str, metadata: str) -> None:
    text = DOCS_PATH.read_text()
    if BENCH_START not in text or BENCH_END not in text:
        raise RuntimeError(
            f"Could not find {BENCH_START!r}/{BENCH_END!r} markers in {DOCS_PATH}. "
            f"Add the benchmark section stub first."
        )
    start = text.index(BENCH_START) + len(BENCH_START)
    end = text.index(BENCH_END)
    block = (
        "\n<!-- generated by tests/bench_mps.py; do not edit by hand -->\n"
        f"{metadata}\n\n{table_md}\n"
    )
    new_text = text[:start] + block + text[end:]
    DOCS_PATH.write_text(new_text)
    print(f"Wrote benchmark table to {DOCS_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.insert(0, torch.device("mps"))
    else:
        print("[warn] MPS unavailable; reporting CPU only.")
    return devices


def _run_one(kernel: KernelBench, scene: Scene, reps: int, warmup: int) -> Optional[float]:
    try:
        ms = kernel.run(scene, reps, warmup)
    except Exception as exc:  # pragma: no cover - surfaced in output
        print(f"  [{kernel.display}] FAILED on {scene.device}: {exc}")
        return None
    if math.isnan(ms):
        return None
    return ms


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MPS kernel benchmark harness")
    parser.add_argument("--n", type=int, default=100_000, help="number of Gaussians")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--kernels", type=str, default=None,
        help="comma-separated kernel keys (default: all). Keys: "
             + ", ".join(k.key for k in KERNELS),
    )
    parser.add_argument(
        "--write-docs", action="store_true",
        help="Overwrite the benchmark section in docs/DEV.md",
    )
    args = parser.parse_args(argv)

    if args.kernels:
        selected_keys = [s.strip() for s in args.kernels.split(",") if s.strip()]
        unknown = [k for k in selected_keys if k not in KERNELS_BY_KEY]
        if unknown:
            parser.error(f"unknown kernel keys: {unknown}")
        selected = [KERNELS_BY_KEY[k] for k in selected_keys]
    else:
        selected = KERNELS

    devices = _resolve_devices()
    rows: list[Row] = []

    for kernel in selected:
        if kernel.slow:
            n_eff = min(args.n, SLOW_KERNEL_N_CAP)
            w_eff = min(args.width, SLOW_KERNEL_WH_CAP)
            h_eff = min(args.height, SLOW_KERNEL_WH_CAP)
            suffix = " (reduced)" if n_eff < args.n else ""
        else:
            n_eff, w_eff, h_eff, suffix = args.n, args.width, args.height, ""

        print(f"\n>> {kernel.display}  (N={n_eff:,}, {w_eff}x{h_eff}{suffix})")
        timings: dict[str, Optional[float]] = {}
        for device in devices:
            scene = build_scene(n_eff, w_eff, h_eff, args.seed, device)
            ms = _run_one(kernel, scene, args.reps, args.warmup)
            timings[device.type] = ms
            if ms is None:
                print(f"  {device.type}: failed")
            else:
                print(f"  {device.type}: {ms:,.3f} ms (median of {args.reps})")

        rows.append(
            Row(
                name=kernel.display,
                n=n_eff,
                wh=f"{w_eff}×{h_eff}",
                mps_ms=timings.get("mps"),
                cpu_ms=timings.get("cpu"),
            )
        )

    table = render_table(rows)
    print("\n" + table)

    if args.write_docs:
        import platform
        meta = (
            f"Hardware: {platform.platform()} · torch {torch.__version__}\n"
            f"Config: reps={args.reps}, warmup={args.warmup}, seed={args.seed}"
        )
        write_docs(table, meta)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
