# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from gsplat.lod import (
    build_hierarchy,
    build_hspt,
    compute_render_set,
    extract_frustum_planes,
    spheres_in_frustum,
)


def _random_splats(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return {
        "means": torch.randn(n, 3, generator=g) * 2.0,
        "scales": (torch.rand(n, 3, generator=g) * 0.3 - 2.5),
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "opacities": torch.full((n,), 2.0),
        "sh0": torch.randn(n, 1, 3, generator=g) * 0.3,
        "shN": torch.zeros(n, 0, 3),
    }


# --- Frustum -----------------------------------------------------------------


def test_frustum_planes_and_sphere_in():
    # Identity camera at origin looking down +z, focal=100, 200x200.
    viewmat = torch.eye(4)
    K = torch.tensor([[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]])
    planes = extract_frustum_planes(viewmat, K, 200, 200, near=0.1, far=100.0)
    # Point directly in front of the camera should be inside.
    mu = torch.tensor([[0.0, 0.0, 5.0]])
    radius = torch.tensor([0.1])
    assert bool(spheres_in_frustum(mu, radius, planes).item())
    # Point well behind the camera should be outside.
    mu = torch.tensor([[0.0, 0.0, -5.0]])
    assert not bool(spheres_in_frustum(mu, radius, planes).item())
    # Large sphere behind camera but overlapping the near plane -> inside.
    mu = torch.tensor([[0.0, 0.0, -0.05]])
    radius = torch.tensor([10.0])
    assert bool(spheres_in_frustum(mu, radius, planes).item())


def test_frustum_extreme_offscreen():
    viewmat = torch.eye(4)
    K = torch.tensor([[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]])
    planes = extract_frustum_planes(viewmat, K, 200, 200, near=0.1, far=100.0)
    # Point 100 meters to the side at z=1 is way outside the frustum.
    mu = torch.tensor([[100.0, 0.0, 1.0]])
    radius = torch.tensor([0.1])
    assert not bool(spheres_in_frustum(mu, radius, planes).item())


# --- HSPT --------------------------------------------------------------------


def test_hspt_size_threshold_infinite_size():
    # size = +inf means every node has volume <= size; root's children are
    # all SPT roots (so the whole thing below root is two SPTs).
    sp = _random_splats(16, seed=3)
    h = build_hierarchy(sp)
    hs = build_hspt(h, size=float("inf"))
    # Upper set should be just the root.
    assert hs.n_upper == 1
    # If root is internal, expect 2 SPTs (its children). If root is a leaf
    # (1-node tree), we have 1 SPT but this test uses n=16 so root is internal.
    assert hs.n_spts == 2
    # The entries in the two SPTs together cover all 2N-1 nodes.
    assert hs.n_entries == h.n_total - 1  # the root itself is in upper, not SPTs


def test_hspt_size_threshold_zero():
    # size = 0 forces every non-leaf child into upper (nothing is <= 0).
    # Only actual leaves can be SPT roots (leaves trivially go to SPTs).
    sp = _random_splats(16, seed=4)
    h = build_hierarchy(sp)
    hs = build_hspt(h, size=0.0)
    n_internal = h.n_total - h.n_leaves
    # Upper contains all internals.
    assert hs.n_upper == n_internal
    # Each leaf is its own SPT.
    assert hs.n_spts == h.n_leaves
    # Each SPT has exactly one entry (the leaf).
    assert hs.n_entries == h.n_leaves


def test_hspt_entries_sorted_descending_by_size_parent():
    sp = _random_splats(32, seed=5)
    h = build_hierarchy(sp)
    hs = build_hspt(h, size=1e-3)
    for s in range(hs.n_spts):
        beg, end = hs.spt_range(s)
        md = hs.spt_entries_size_parent[beg:end]
        if md.numel() > 1:
            diffs = md[:-1] - md[1:]
            assert (diffs >= -1e-6).all(), f"spt {s} not sorted descending"


# --- Cut --------------------------------------------------------------------


def _identity_camera():
    viewmat = torch.eye(4)
    K = torch.tensor([[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]])
    width, height = 200, 200
    return viewmat, K, width, height


def test_cut_large_T_returns_root():
    # Very large T => m_d(root) is huge => even at modest distances we ALWAYS
    # have d >= m_d only at root. Wait, m_d grows with T, so d < m_d for all
    # inner nodes including root at big T => BFS descends everywhere => cut
    # contains leaves. Let's instead test the opposite: T very small => m_d
    # tiny everywhere, cut stops at root.
    sp = _random_splats(32, seed=6)
    # Move all Gaussians far in front of the camera so they are visible.
    sp["means"] = sp["means"] + torch.tensor([0.0, 0.0, 10.0])
    h = build_hierarchy(sp)
    hs = build_hspt(h, size=float("inf"))

    viewmat, K, W, H = _identity_camera()
    camera_pos = torch.zeros(3)

    # T very small: m_d tiny, d >= m_d satisfied at root -> cut = {root}.
    rs = compute_render_set(
        hs, camera_pos, viewmat, K, W, H, T=1e-8, frustum_radius_mult=3.0
    )
    assert rs.upper_cut_ids.numel() == 1
    assert int(rs.upper_cut_ids[0]) == h.root


def test_cut_huge_T_returns_leaves():
    # T huge: m_d huge everywhere, d < m_d always, descend to leaves. SPT
    # cut: binary search keeps prefix where md_parent > d. At huge T, md_parent
    # >> any finite d, so all entries are kept, which is all leaves + internals
    # of the SPTs. We verify that the render set includes EVERY leaf.
    sp = _random_splats(32, seed=7)
    sp["means"] = sp["means"] + torch.tensor([0.0, 0.0, 10.0])
    h = build_hierarchy(sp)
    hs = build_hspt(h, size=float("inf"))

    viewmat, K, W, H = _identity_camera()
    camera_pos = torch.zeros(3)

    rs = compute_render_set(
        hs, camera_pos, viewmat, K, W, H, T=1e8, frustum_radius_mult=3.0
    )
    # Every leaf must appear in the render set.
    leaf_ids = torch.nonzero(h.is_leaf, as_tuple=False).squeeze(-1)
    rendered = set(rs.node_ids.tolist())
    for lid in leaf_ids.tolist():
        assert lid in rendered, f"leaf {lid} missing from render set"


def test_cut_monotonic_in_T():
    # As T increases, #Gaussians in render set should be (weakly) monotonic up.
    sp = _random_splats(64, seed=8)
    sp["means"] = sp["means"] + torch.tensor([0.0, 0.0, 5.0])
    h = build_hierarchy(sp)
    hs = build_hspt(h, size=1e-4)

    viewmat, K, W, H = _identity_camera()
    camera_pos = torch.zeros(3)

    counts = []
    for T in [1e-6, 1e-3, 1e-1, 1.0, 100.0]:
        # Re-precompute upper metrics for each T.
        hs.upper_m_d = None
        hs.upper_max_scale = None
        hs.upper_mu = None
        rs = compute_render_set(hs, camera_pos, viewmat, K, W, H, T=T)
        counts.append(rs.node_ids.numel())

    for a, b in zip(counts[:-1], counts[1:]):
        assert b >= a, f"non-monotonic counts: {counts}"


def test_cut_frustum_culling_reduces_set():
    # Split Gaussians into "in frustum" (front) and "way out" (behind camera).
    g = torch.Generator().manual_seed(9)
    n = 32
    means_front = torch.randn(n, 3, generator=g) * 1.0 + torch.tensor([0.0, 0.0, 5.0])
    means_behind = torch.randn(n, 3, generator=g) * 1.0 + torch.tensor([0.0, 0.0, -20.0])
    sp = _random_splats(2 * n, seed=10)
    sp["means"] = torch.cat([means_front, means_behind], dim=0)

    h = build_hierarchy(sp)
    hs = build_hspt(h, size=1e-4)

    viewmat, K, W, H = _identity_camera()
    camera_pos = torch.zeros(3)

    # Pick a moderately large T so the BFS descends past the root.
    rs_culled = compute_render_set(hs, camera_pos, viewmat, K, W, H, T=100.0)
    rs_uncull = compute_render_set(
        hs, camera_pos, viewmat, K, W, H, T=100.0, skip_frustum=True
    )
    assert rs_culled.node_ids.numel() <= rs_uncull.node_ids.numel()
