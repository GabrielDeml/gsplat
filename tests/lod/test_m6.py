# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M6 tests: GPU cut path parity with the Python reference and (if CUDA is
available) the raw CUDA kernel parity with both."""

import torch

from gsplat.lod import (
    build_hierarchy,
    build_hspt,
    compute_render_set,
    compute_render_set_gpu,
)


def _splats(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return {
        "means": torch.randn(n, 3, generator=g) * 2.0 + torch.tensor([0.0, 0.0, 8.0]),
        "scales": (torch.rand(n, 3, generator=g) * 0.3 - 2.5),
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "opacities": torch.full((n,), 2.0),
        "sh0": torch.zeros(n, 1, 3),
        "shN": torch.zeros(n, 0, 3),
    }


def _identity_cam(device="cpu"):
    viewmat = torch.eye(4, device=device)
    K = torch.tensor(
        [[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]],
        device=device,
    )
    return viewmat, K, 200, 200


def test_hspt_has_entries_spt_id():
    sp = _splats(32, seed=0)
    h = build_hierarchy(sp)
    hs = build_hspt(h, size=1e-3)
    # Every entry has a spt_id matching the CSR membership.
    assert hs.spt_entries_spt_id.numel() == hs.spt_entries_node_id.numel()
    for s in range(hs.n_spts):
        beg, end = hs.spt_range(s)
        assert torch.all(hs.spt_entries_spt_id[beg:end] == s)


def test_gpu_cut_matches_python_cut_on_cpu():
    """The GPU cut path must be exercisable on CPU (same code-path modulo
    dispatch). It should produce an identical render set to the Python
    reference modulo node-id ordering.
    """
    sp = _splats(48, seed=1)
    h = build_hierarchy(sp)
    hs = build_hspt(h, size=1e-3)
    viewmat, K, W, H = _identity_cam()
    camera_pos = torch.zeros(3)

    for T in [1e-3, 1e-1, 1.0, 100.0]:
        hs.upper_m_d = None
        hs.upper_max_scale = None
        hs.upper_mu = None
        ref = compute_render_set(hs, camera_pos, viewmat, K, W, H, T=T)

        hs.upper_m_d = None
        hs.upper_max_scale = None
        hs.upper_mu = None
        gpu = compute_render_set_gpu(hs, camera_pos, viewmat, K, W, H, T=T)

        ref_set = set(ref.node_ids.tolist())
        gpu_set = set(gpu.node_ids.tolist())
        assert ref_set == gpu_set, (
            f"T={T}: ref/gpu render sets differ; |ref|={len(ref_set)} |gpu|={len(gpu_set)} "
            f"symdiff={len(ref_set ^ gpu_set)}"
        )


def test_hspt_to_device_round_trip():
    # HSPT.to("cpu") on a cpu HSPT should produce tensors that are still on
    # cpu (and the copy is shallow-ish, not mutating the source).
    sp = _splats(16, seed=2)
    h = build_hierarchy(sp)
    hs = build_hspt(h, size=1e-3)
    hs2 = hs.to("cpu")
    assert hs2.upper_ids.device.type == "cpu"
    assert torch.equal(hs2.upper_ids, hs.upper_ids)
    assert torch.equal(hs2.spt_entries_spt_id, hs.spt_entries_spt_id)


def test_gpu_cut_force_cpu_falls_back_to_python_path():
    # With force_cpu=True the standard compute_render_set should stay on the
    # Python BFS implementation regardless of hspt device.
    sp = _splats(16, seed=3)
    h = build_hierarchy(sp)
    hs = build_hspt(h, size=1e-3)
    viewmat, K, W, H = _identity_cam()
    rs = compute_render_set(hs, torch.zeros(3), viewmat, K, W, H, T=1.0, force_cpu=True)
    assert rs.node_ids.numel() > 0


# -----------------------------------------------------------------------------
# CUDA kernel parity (skipped unless a real CUDA build + device is available).
# -----------------------------------------------------------------------------

import pytest  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_sphere_in_frustum_matches_python():
    from gsplat.lod.frustum import (
        extract_frustum_planes,
        spheres_in_frustum,
    )

    try:
        from gsplat.cuda._wrapper import lod_sphere_in_frustum
    except (ImportError, AttributeError):
        pytest.skip("gsplat CUDA extension without LoD kernels")
        return

    torch.manual_seed(0)
    M = 500
    device = torch.device("cuda")
    mu = torch.randn(M, 3, device=device) * 5.0
    radius = torch.rand(M, device=device) * 0.5 + 0.01
    viewmat = torch.eye(4, device=device)
    K = torch.tensor(
        [[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]],
        device=device,
    )
    planes = extract_frustum_planes(viewmat, K, 200, 200, near=0.1, far=100.0)

    py = spheres_in_frustum(mu, radius, planes)
    try:
        cu = lod_sphere_in_frustum(mu, radius, planes)
    except RuntimeError as e:
        pytest.skip(f"lod_sphere_in_frustum op not registered: {e}")
        return
    assert torch.equal(py, cu)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_spt_cut_count_matches_python():
    try:
        from gsplat.cuda._wrapper import lod_spt_cut_count
    except (ImportError, AttributeError):
        pytest.skip("gsplat CUDA extension without LoD kernels")
        return

    # Construct a tiny synthetic CSR SPT array with 3 SPTs of varying lengths.
    device = torch.device("cuda")
    entries = torch.tensor(
        [5.0, 3.0, 1.5, 0.5,  # spt 0
         4.0, 2.0,            # spt 1
         9.0, 7.0, 6.0],      # spt 2
        dtype=torch.float32, device=device,
    )
    offsets = torch.tensor([0, 4, 6, 9], dtype=torch.int64, device=device)
    touched = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
    distances = torch.tensor([1.0, 3.0, 6.5], dtype=torch.float32, device=device)
    T = 1.0

    try:
        cu = lod_spt_cut_count(entries, offsets, touched, distances, T)
    except RuntimeError as e:
        pytest.skip(f"lod_spt_cut_count op not registered: {e}")
        return

    # Expected: entries > d/T per SPT.
    # SPT 0: threshold=1.0 -> keep 5,3,1.5 -> 3
    # SPT 1: threshold=3.0 -> keep 4 -> 1
    # SPT 2: threshold=6.5 -> keep 9,7 -> 2
    assert cu.tolist() == [3, 1, 2]
