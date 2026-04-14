# MPS Backend Implementation Plan

**Scope:** full porting plan, Tiers 0–4 (infra → core 3DGS → extended 3DGS → 2DGS → lidar).
**Audience:** a swarm of agents, each picking up a single item and executing it with no prior context.

---

## Context

`gsplat/mps/` currently mirrors the CUDA API in `_wrapper.py`, but every kernel
delegates to pure-PyTorch references in `gsplat/cuda/_torch_impl*.py`. Only
`csrc/bootstrap.metal` exists, to prove runtime `torch.mps.compile_shader`
works. Two functions raise `NotImplementedError` (`isect_tiles_lidar`,
`rasterize_to_indices_in_range_2dgs`) and two feature flags return False
(`has_camera_wrappers`, `has_reloc`).

Goal: replace every pure-PyTorch fallback with a native Metal kernel and fill
feature gaps, reaching CUDA-equivalent behavior on Apple Silicon.

---

## How to execute a checklist item (read this before starting any task)

Each leaf checkbox below is scoped so one agent with zero prior context can
execute it independently. For every kernel task:

1. **Read the CUDA reference** file listed under "Reference".
2. **Read the pure-PyTorch oracle** file listed under "Oracle" (when available).
3. **Read the Python wrapper** at the line number listed under "Wire-in point"
   in `gsplat/mps/_wrapper.py` to understand the I/O contract.
4. **Write the `.metal` file** into `gsplat/mps/csrc/`. The build system
   (`gsplat/mps/build.py`) auto-discovers, concatenates, hashes, and JIT-compiles
   everything in that directory; no build changes needed.
5. **Wire into `_wrapper.py`**: replace the existing pure-PyTorch fallback
   block with `_C.<kernel_name>(...)` gated by a `_C is not None` check.
   Keep the pure-PyTorch path reachable via a try/except during rollout; drop
   the fallback in a follow-up PR once stable.
6. **Add tests** that compare forward output and (where applicable) backward
   gradients to the oracle, using the tolerance patterns already in
   `tests/test_basic.py`.
7. **Run `bash run_mps_tests.sh`** and confirm no regressions.
8. **Check the box.**

### Shared ground rules

- Mirror CUDA file names where possible (e.g. `RasterizeToPixels3DGSFwd.cu` →
  `RasterizeToPixels3DGSFwd.metal`).
- Tile size, threadgroup layout, and shared-memory budget are per-kernel design
  decisions — document the choice in a short comment at the top of the kernel.
- Atomic float add: use `atomic_fetch_add_explicit` on `atomic<float>` in
  Metal 3+. Helpers go in `csrc/common.metal` (Tier 0).

---

## Tier 0 — Infrastructure & test scaffolding

These unblock everything else. Do these first, in order.

- [x] **T0.1 — MPS device pytest fixture**
    - [x] Open `tests/conftest.py` (create if missing)
    - [x] Add a `device` fixture that parametrizes `cuda`, `mps`, `cpu` with
          skip markers (`torch.cuda.is_available()`,
          `torch.backends.mps.is_available()`)
    - [x] Document usage in a comment block at the top
- [x] **T0.2 — Forward parity helper**
    - [x] In `tests/test_mps_backend.py` add `assert_mps_matches_reference(
          mps_out, ref_out, rtol, atol)` that wraps `torch.testing.assert_close`
          with per-tensor tolerances
- [x] **T0.3 — Backward parity helper**
    - [x] Add `assert_mps_grads_match_reference(inputs, mps_fn, ref_fn, ...)`
          that runs `torch.autograd.grad` on both paths and compares
- [ ] **T0.4 — Shared Metal utilities**
    - [ ] Create `gsplat/mps/csrc/common.metal` (will be concatenated by the
          build bundle — keep symbols prefixed `gsplat_`)
    - [ ] Add `quat_to_rotmat(float4 q) -> float3x3`
    - [ ] Add `mat3_inverse(float3x3 m) -> float3x3`
    - [ ] Add `mat2_inverse(float2x2 m) -> float2x2`
    - [ ] Add `atomic_add_float(device atomic<float>* addr, float val)`
    - [ ] Add tile-size constants (`TILE_WIDTH=16`, `TILE_HEIGHT=16`) matching
          CUDA defaults
- [ ] **T0.5 — Documentation**
    - [ ] Add an "MPS porting status" table to `docs/DEV.md` with one row per
          kernel, columns: Kernel / Status / PR
    - [ ] Document the kernel-add workflow steps (matching "How to execute
          a checklist item" above)
- [ ] **T0.6 — Benchmark harness**
    - [ ] Add `tests/bench_mps.py` that times each kernel vs the
          pure-PyTorch path on a fixed 100k-Gaussian scene
    - [ ] Print a table; write results into `docs/DEV.md` on each tier
          completion
- [ ] **T0.7 — CI coverage**
    - [ ] Update `run_mps_tests.sh` so new tests are picked up automatically
          (confirm the `pytest tests/` invocation globs new files)

---

## Tier 1 — Core 3DGS pipeline

Order within Tier 1 matters: later items depend on earlier ones.

### T1.1 — quat/scale → covariance/precision

- **Reference:** `gsplat/cuda/csrc/QuatScaleToCovarCUDA.cu`
- **Oracle:** `gsplat/cuda/_torch_impl._quat_scale_to_covar_preci`
- **Wire-in point:** `gsplat/mps/_wrapper.py:~375` (`quat_scale_to_covar_preci`)

- [ ] Write `csrc/QuatScaleToCovarPreciFwd.metal`
- [ ] Write `csrc/QuatScaleToCovarPreciBwd.metal`
- [ ] Wire forward into `_wrapper.quat_scale_to_covar_preci`
- [ ] Wire backward into the corresponding `torch.autograd.Function`
- [ ] Test: forward parity (dense + triu outputs)
- [ ] Test: backward parity (grads to quats, scales)
- [ ] Benchmark row added

### T1.2 — Spherical harmonics

- **Reference:** `gsplat/cuda/csrc/SphericalHarmonicsCUDA.cu`
- **Oracle:** `gsplat/cuda/_torch_impl._spherical_harmonics`,
  `_eval_sh_bases_fast`
- **Wire-in point:** `gsplat/mps/_wrapper.py:~342`

- [ ] Write `csrc/SphericalHarmonicsFwd.metal` (support orders 0–4, masked)
- [ ] Write `csrc/SphericalHarmonicsBwd.metal`
- [ ] Wire into `_wrapper.spherical_harmonics`
- [ ] Test: forward parity for each order 0..4, with and without mask
- [ ] Test: backward parity (grads to dirs, coeffs)
- [ ] Benchmark row added

### T1.3 — EWA projection (fused, dense)

- **Reference:** `gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu`
- **Oracle:** `gsplat/cuda/_torch_impl._fully_fused_projection`
- **Wire-in point:** `gsplat/mps/_wrapper.py` `fully_fused_projection` (dense path)
- **Depends on:** T0.4 (quat_to_rotmat util)

- [ ] Write `csrc/ProjectionEWA3DGSFusedFwd.metal` — all camera models:
  pinhole, ortho, fisheye; compute means2d, conics, depths, radii, compensation
- [ ] Write `csrc/ProjectionEWA3DGSFusedBwd.metal`
- [ ] Wire forward + backward (dense path only; packed is T2.1)
- [ ] Test: forward parity per camera model
- [ ] Test: backward parity (grads to means, quats, scales, viewmats, Ks)
- [ ] Test: radii/compensation match CUDA within single-ulp
- [ ] Benchmark row added

### T1.4 — Tile intersection (3DGS)

- **Reference:** `gsplat/cuda/csrc/IntersectTile.cu`
- **Oracle:** `gsplat/cuda/_torch_impl._isect_tiles`
- **Wire-in point:** `gsplat/mps/_wrapper.py:~635` (`isect_tiles`)

- [ ] Write `csrc/IntersectTile.metal`
    - [ ] Per-Gaussian tile enumeration (AABB of projected ellipse ∩ tile grid)
    - [ ] 64-bit bit-packed `isect_ids = (img_id, tile_id, depth_bits)` matching
          the packing used in `_isect_tiles` exactly
    - [ ] Optional depth sort — use `torch.sort` on the host side if a Metal
          sort is not trivially available; note the design decision in a
          kernel header comment
- [ ] Wire into `_wrapper.isect_tiles`
- [ ] Test: `tiles_per_gauss` matches oracle
- [ ] Test: `isect_ids` packing matches oracle bit-for-bit
- [ ] Test: `flatten_ids` matches after deterministic sort
- [ ] Benchmark row added

### T1.5 — Tile offset encoding

- **Reference:** `gsplat/cuda/csrc/Intersect.cpp` (`launch_intersect_offset_kernel`)
- **Oracle:** `gsplat/cuda/_torch_impl._isect_offset_encode`
- **Wire-in point:** `gsplat/mps/_wrapper.py:~807` (`isect_offset_encode`)

- [ ] Write `csrc/IntersectOffset.metal` OR implement in pure MPS if a
      single-pass scan isn't worth a kernel — document the choice
- [ ] Wire into `_wrapper.isect_offset_encode`
- [ ] Test: offsets match oracle for varied tile counts
- [ ] Benchmark row added

### T1.6 — Rasterize-to-pixels (forward + backward)

This is the largest single task. Subdivide aggressively.

- **Reference fwd:** `gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu`
- **Reference bwd:** `gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu`
- **Oracle:** `gsplat/cuda/_torch_impl._rasterize_to_pixels`
- **Wire-in point:** `gsplat/mps/_wrapper.py:~1709` (`_RasterizeToPixels`)
- **Depends on:** T1.4, T1.5, T0.4

- [ ] **T1.6a** — Validate Metal threadgroup memory budget on target Apple
      GPU (query `[MTLDevice maxThreadgroupMemoryLength]` equivalent; plan
      around 32KB typical)
- [ ] **T1.6b** — Validate atomic float add performance and correctness on
      device memory (microbench)
- [ ] **T1.6c** — Pick tile size (default 16×16) and document in kernel header
- [ ] **T1.6d** — Write `csrc/RasterizeToPixels3DGSFwd.metal`
    - [ ] Tile-batched Gaussian loading into threadgroup memory
    - [ ] Per-pixel alpha blend with early termination on `T < 1e-4`
    - [ ] Multi-channel color support (test at COLOR_DIM ∈ {3, 32})
    - [ ] Background / mask support
    - [ ] Output: render, alpha, last_ids
- [ ] **T1.6e** — Write `csrc/RasterizeToPixels3DGSBwd.metal`
    - [ ] Gradients: means2d, conics, colors, opacities
    - [ ] absgrad mode (writes |grad| into a separate buffer)
- [ ] **T1.6f** — Wire forward into `_RasterizeToPixels.forward`
- [ ] **T1.6g** — Wire backward into `_RasterizeToPixels.backward`
- [ ] **T1.6h** — Test: forward parity for a small fixed scene (CPU-oracle)
- [ ] **T1.6i** — Test: forward parity vs CUDA on a larger scene (cross-device)
- [ ] **T1.6j** — Test: backward parity (means2d grads)
- [ ] **T1.6k** — Test: backward parity (conics grads)
- [ ] **T1.6l** — Test: backward parity (colors + opacities grads)
- [ ] **T1.6m** — Test: absgrad mode produces same values as `|grad|`
- [ ] **T1.6n** — Benchmark row added; goal is within 3× of CUDA on an
      M-series GPU

### T1.7 — Rasterize-to-indices in range

- **Reference:** `gsplat/cuda/csrc/RasterizeToIndices3DGS.cu`
- **Oracle:** pure-PyTorch loop in
  `gsplat/mps/_wrapper.rasterize_to_indices_in_range` (current fallback)
- **Wire-in point:** `gsplat/mps/_wrapper.py:~1274`
- **Why priority:** explicitly the largest perf bottleneck in the current
  pure-PyTorch MPS path.

- [ ] Write `csrc/RasterizeToIndices3DGS.metal`
- [ ] Wire into `_wrapper.rasterize_to_indices_in_range`
- [ ] Test: index set equality vs PyTorch reference (order-insensitive)
- [ ] Test: stability across transmittance range values
- [ ] Benchmark row added

### T1.8 — Adam optimizer step

- **Reference:** `gsplat/cuda/csrc/AdamCUDA.cu`
- **Oracle:** `torch.optim.Adam` applied manually to same tensors
- **Wire-in point:** `gsplat/mps/_wrapper.py:~326` (`adam`)

- [ ] Write `csrc/Adam.metal` (fused in-place update with optional valid mask)
- [ ] Wire into `_wrapper.adam`
- [ ] Test: one step matches `torch.optim.Adam` step within fp32 noise
- [ ] Test: masked variant only updates selected rows
- [ ] Benchmark row added

### T1.9 — Tier 1 acceptance

- [ ] `run_mps_tests.sh` passes on Apple Silicon with all Tier-1 native
      kernels active (no PyTorch fallback taken)
- [ ] Benchmark table in `docs/DEV.md` updated with Tier-1 speedups vs the
      pure-PyTorch baseline
- [ ] Train a small 3DGS scene to convergence on MPS; record final PSNR vs
      CUDA baseline in `docs/DEV.md` (target: within 0.5 dB)

---

## Tier 2 — 3DGS extended features

All Tier-2 tasks can be parallelized after T1.3 ships.

### T2.1 — Packed EWA projection

- **Reference:** `gsplat/cuda/csrc/ProjectionEWA3DGSPacked.cu`
- **Wire-in point:** `_wrapper.fully_fused_projection` (packed path)

- [ ] Write `csrc/ProjectionEWA3DGSPackedFwd.metal`
- [ ] Write `csrc/ProjectionEWA3DGSPackedBwd.metal`
- [ ] Wire packed path in `_wrapper.fully_fused_projection`
- [ ] Test: forward + backward parity
- [ ] Benchmark row added

### T2.2 — Rasterize from world (eval3d)

- **Reference fwd:** `gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGSFwd.cu`
- **Reference bwd:** `gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGSBwd.cu`
- **Oracle:** `gsplat/cuda/_torch_impl_eval3d._rasterize_to_pixels_eval3d`
- **Wire-in point:** `gsplat/mps/_wrapper.py:~1839` (`_RasterizeToPixelsEval3D`)

- [ ] Write `csrc/RasterizeToPixelsFromWorld3DGSFwd.metal`
- [ ] Write `csrc/RasterizeToPixelsFromWorld3DGSBwd.metal`
- [ ] Wire forward into `_RasterizeToPixelsEval3D.forward`
- [ ] Wire backward into `_RasterizeToPixelsEval3D.backward`
- [ ] Resolve the `NotImplementedError` in
      `rasterize_to_pixels_eval3d_extra` for packed-mode + colors
- [ ] Tests: forward + backward parity for each camera-model flag
- [ ] Benchmark row added

### T2.3 — Unscented-transform projection (forward only, matches CUDA)

- **Reference:** `gsplat/cuda/csrc/ProjectionUT3DGSFused.cu`
- **Oracle:** `gsplat/cuda/_torch_impl_ut._fully_fused_projection_with_ut`
- **Wire-in point:** `gsplat/mps/_wrapper.py:~1597`

- [ ] Write `csrc/ProjectionUT3DGSFused.metal` (sigma-point generation, per-point
      camera distortion eval, covariance reconstruction)
- [ ] Wire into `_wrapper.fully_fused_projection_with_ut`
- [ ] Test: forward parity per camera model (ftheta, lidar, bivariate,
      rolling-shutter)
- [ ] Benchmark row added

### T2.4 — Camera wrappers + external distortion

This is a large chunk; split by camera model.

- **References:** `CameraWrappers.cu`, `ExternalDistortionWrappers.cu`
- **Wire-in point:** `_wrapper.create_camera_model` (line ~205) and
  `has_camera_wrappers()` flag in `gsplat/mps/__init__.py`

- [ ] **T2.4a** — Port pinhole / ortho / fisheye distortion evaluators
- [ ] **T2.4b** — Port ftheta
- [ ] **T2.4c** — Port bivariate (windshield-style) distortion
- [ ] **T2.4d** — Port rolling-shutter handling
- [ ] **T2.4e** — Flip `has_camera_wrappers()` → `True` once T2.4a–d land
- [ ] Per-model parity tests against CUDA

### T2.5 — MCMC relocation

- **Reference:** `gsplat/cuda/csrc/RelocationCUDA.cu`
- **Wire-in point:** `_wrapper.relocation` + `has_reloc()` in
  `gsplat/mps/__init__.py`

- [ ] Write `csrc/Relocation.metal` (precomputed binomial table, per-Gaussian
      opacity power law)
- [ ] Wire into `_wrapper.relocation`
- [ ] Flip `has_reloc()` → `True`
- [ ] Test: parity with CUDA on fixed inputs
- [ ] Benchmark row added

### T2.6 — Tier 2 acceptance

- [ ] `pytest tests/test_rasterization.py` passes on MPS (minus distributed
      and lidar)
- [ ] Updated benchmark table in `docs/DEV.md`

---

## Tier 3 — 2DGS pipeline

No pure-PyTorch references exist. Oracle is CUDA output on the same inputs —
test on a CUDA host and capture reference tensors, OR write a slow
PyTorch oracle alongside each task.

### T3.1 — 2DGS fused projection

- **Reference:** `gsplat/cuda/csrc/Projection2DGSFused.cu`, `Projection2DGS.cuh`
- **Wire-in point:** `_wrapper.fully_fused_projection_2dgs`

- [ ] **T3.1a** — Port `Projection2DGS.cuh` helpers into
      `csrc/projection_2dgs_common.metal`
- [ ] **T3.1b** — Write `csrc/Projection2DGSFusedFwd.metal`
- [ ] **T3.1c** — Write `csrc/Projection2DGSFusedBwd.metal`
- [ ] **T3.1d** — Wire forward + backward
- [ ] **T3.1e** — Tests: forward parity vs captured CUDA reference
- [ ] **T3.1f** — Tests: backward parity vs captured CUDA reference
- [ ] Benchmark row added

### T3.2 — 2DGS packed projection

- **Reference:** `gsplat/cuda/csrc/Projection2DGSPacked.cu`

- [ ] Write `csrc/Projection2DGSPackedFwd.metal`
- [ ] Write `csrc/Projection2DGSPackedBwd.metal`
- [ ] Wire packed path
- [ ] Tests: fwd + bwd parity
- [ ] Benchmark row added

### T3.3 — 2DGS rasterization

- **Reference fwd:** `gsplat/cuda/csrc/RasterizeToPixels2DGSFwd.cu`
- **Reference bwd:** `gsplat/cuda/csrc/RasterizeToPixels2DGSBwd.cu`
- **Wire-in point:** `_wrapper.rasterize_to_pixels_2dgs` (line ~2754)

- [ ] Write `csrc/RasterizeToPixels2DGSFwd.metal`
- [ ] Write `csrc/RasterizeToPixels2DGSBwd.metal`
- [ ] Wire forward
- [ ] Wire backward
- [ ] Tests: render + grads parity on a 2DGS reference scene
- [ ] Benchmark row added

### T3.4 — 2DGS rasterize-to-indices (remove `NotImplementedError`)

- **Reference:** `gsplat/cuda/csrc/RasterizeToIndices2DGS.cu`
- **Wire-in point:** `_wrapper.rasterize_to_indices_in_range_2dgs` (line ~2888)

- [ ] Write `csrc/RasterizeToIndices2DGS.metal`
- [ ] Delete the `NotImplementedError` branch and wire the kernel in
- [ ] Tests
- [ ] Benchmark row added

### T3.5 — Tier 3 acceptance

- [ ] `pytest tests/test_2dgs.py` passes on MPS
- [ ] Benchmark table updated

---

## Tier 4 — Lidar

Currently excluded by `run_mps_tests.sh` (`-k "not lidar"`). Do last.

### T4.1 — Lidar tile intersection

- **Reference:** `gsplat/cuda/csrc/IntersectTileLidar.cu`
- **Wire-in point:** `_wrapper.isect_tiles_lidar` (line ~734, currently
  `NotImplementedError`)

- [ ] Write `csrc/IntersectTileLidar.metal`
- [ ] Delete the `NotImplementedError` branch and wire the kernel in
- [ ] Tests: lidar parity vs CUDA
- [ ] Benchmark row added

### T4.2 — Tier 4 acceptance

- [ ] Remove `-k "not lidar"` filter from `run_mps_tests.sh`
- [ ] Full `pytest tests/` suite green on MPS (excluding distributed)
- [ ] Final benchmark table in `docs/DEV.md`

---

## Critical files (reference map)

| Path | Role |
|---|---|
| `gsplat/mps/csrc/*.metal` | New Metal kernel sources — auto-discovered by `build.py` |
| `gsplat/mps/_wrapper.py` | Per-kernel wire-in points (line numbers cited above) |
| `gsplat/mps/__init__.py` | `has_camera_wrappers()` / `has_reloc()` flags |
| `gsplat/mps/build.py` | Runtime Metal compile + cache (do not modify for new kernels) |
| `gsplat/cuda/csrc/*.cu` | Reference implementations |
| `gsplat/cuda/_torch_impl.py` | Forward/backward oracles (3DGS) |
| `gsplat/cuda/_torch_impl_2dgs.py` | Oracles (2DGS) |
| `gsplat/cuda/_torch_impl_eval3d.py` | Oracles (eval3d) |
| `gsplat/cuda/_torch_impl_ut.py` | Oracles (UT projection) |
| `gsplat/cuda/_wrapper.py` | API contract that MPS wrapper mirrors |
| `tests/test_mps_backend.py` | Where new MPS kernel tests land |
| `tests/bench_mps.py` | Benchmark harness (created in T0.6) |
| `run_mps_tests.sh` | CI entry point |
| `docs/DEV.md` | Status + benchmark table |

---

## Per-kernel verification checklist (apply to every T1/T2/T3/T4 kernel)

- [ ] Forward parity test against the oracle (or captured CUDA reference for 2DGS)
- [ ] Backward parity test via `torch.autograd.grad` (when kernel is differentiable)
- [ ] Cross-device parity test vs CUDA on a host that has both (when available)
- [ ] `run_mps_tests.sh` clean
- [ ] Benchmark row added to `docs/DEV.md`
- [ ] Pure-PyTorch fallback removed once kernel is stable (follow-up PR OK)

---

## Final acceptance (end of Tier 4)

- [ ] All `has_*()` flags return `True` on MPS where they return `True` on CUDA
- [ ] No `NotImplementedError` remains in `gsplat/mps/_wrapper.py`
- [ ] No pure-PyTorch fallback path remains for any rasterization or
      projection kernel
- [ ] Full `pytest tests/` suite green on MPS (excluding distributed)
- [ ] Small scene trained to convergence on MPS within 0.5 dB of CUDA PSNR
- [ ] End-to-end wall-clock speedup table in `docs/DEV.md` shows MPS within
      3× of CUDA on an M-series GPU for the 3DGS pipeline
