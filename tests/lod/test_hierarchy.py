# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from gsplat.lod import GaussianHierarchy, build_hierarchy, merge_pair


def _random_splats(n: int, sh_degree: int = 0, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    K = (sh_degree + 1) ** 2 - 1
    return {
        "means": torch.randn(n, 3, generator=g),
        "scales": (torch.rand(n, 3, generator=g) * 0.5 - 2.5),  # log-scales
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "opacities": torch.full((n,), 2.0),  # sigmoid(2)~0.88
        "sh0": torch.randn(n, 1, 3, generator=g) * 0.3,
        "shN": torch.randn(n, K, 3, generator=g) * 0.1,
    }


def test_single_gaussian():
    sp = _random_splats(1)
    h = build_hierarchy(sp)
    assert h.n_total == 1
    assert h.n_leaves == 1
    assert h.root == 0
    h.validate()


def test_two_gaussians():
    sp = _random_splats(2)
    h = build_hierarchy(sp)
    assert h.n_total == 3  # 2 leaves + 1 internal
    assert h.n_leaves == 2
    assert h.root == 2
    h.validate()
    # Root has the two leaves as children
    assert set([int(h.left[2]), int(h.right[2])]) == {0, 1}


def test_binary_topology_random():
    for n in [3, 5, 16, 100, 500]:
        sp = _random_splats(n)
        h = build_hierarchy(sp)
        h.validate()
        assert h.n_leaves == n
        assert h.n_internal == n - 1
        assert h.n_total == 2 * n - 1


def test_merge_pair_numeric():
    # Two isotropic Gaussians at (0,0,0) and (1,0,0) with s=0.1.
    log_s = torch.log(torch.tensor([0.1, 0.1, 0.1]))
    a = {
        "means": torch.tensor([0.0, 0.0, 0.0]),
        "scales": log_s.clone(),
        "quats": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "opacities": torch.tensor(2.0),
        "sh0": torch.zeros(1, 3),
        "shN": torch.zeros(0, 3),
    }
    b = {
        "means": torch.tensor([1.0, 0.0, 0.0]),
        "scales": log_s.clone(),
        "quats": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "opacities": torch.tensor(2.0),
        "sh0": torch.zeros(1, 3),
        "shN": torch.zeros(0, 3),
    }
    p = merge_pair(a, b)
    # Mean should be at the midpoint.
    assert torch.allclose(p["means"], torch.tensor([0.5, 0.0, 0.0]), atol=1e-5)
    # Opacity union > individual opacities.
    sig_a = torch.sigmoid(torch.tensor(2.0)).item()
    sig_p = torch.sigmoid(p["opacities"]).item()
    assert sig_p > sig_a
    # Parent max-scale must be larger than child (moment-matched; gets spread).
    child_max = torch.exp(a["scales"]).amax().item()
    parent_max = torch.exp(p["scales"]).amax().item()
    assert parent_max > child_max


def test_heap_property_m_d_holds_for_most_nodes():
    """Under moment-matched merging, parent covariance usually (but not
    always) exceeds children's, giving m_d(parent) >= m_d(child).

    When two children have strongly anisotropic but misaligned covariances,
    moment-matching can yield a parent max-eigenvalue smaller than the max
    child's. The paper acknowledges this (A.5 "Unreachable Gaussians") and
    reports <10% of nodes violate the heap. We require at least 90% compliance.
    """
    sp = _random_splats(128, seed=1)
    h = build_hierarchy(sp)
    m_d = h.compute_m_d(T=1.0)
    internal_idx = torch.nonzero(~h.is_leaf, as_tuple=False).squeeze(-1)
    ok = 0
    total = 0
    for i in internal_idx.tolist():
        l = int(h.left[i])
        r = int(h.right[i])
        for c in (l, r):
            total += 1
            if m_d[i].item() >= m_d[c].item() - 1e-6:
                ok += 1
    frac = ok / total
    assert frac >= 0.90, f"heap property held for only {frac*100:.1f}% of parent-child edges"


def test_compute_subtree_size():
    sp = _random_splats(32, seed=2)
    h = build_hierarchy(sp)
    h.compute_subtree_size()
    assert h.subtree_size[h.root].item() == 32
    # Each leaf has subtree size 1.
    leaf_idx = torch.nonzero(h.is_leaf, as_tuple=False).squeeze(-1)
    assert torch.all(h.subtree_size[leaf_idx] == 1)


def test_leaf_splats_round_trip():
    sp = _random_splats(20, seed=3)
    h = build_hierarchy(sp)
    leaves = h.leaf_splats()
    assert leaves["means"].shape == sp["means"].shape
    # Build-order preserves leaf indices [0..N).
    assert torch.allclose(leaves["means"], sp["means"])
    assert torch.allclose(leaves["scales"], sp["scales"])
    assert torch.allclose(leaves["quats"], sp["quats"])
    assert torch.allclose(leaves["opacities"], sp["opacities"])


def test_grow_and_convert():
    sp = _random_splats(4, seed=4)
    h = build_hierarchy(sp)
    old_total = h.n_total
    new_ids = h.grow(2)
    assert h.n_total == old_total + 2
    assert new_ids.tolist() == [old_total, old_total + 1]
    # Mark new ones as leaves (initial state from grow is zero/-1).
    # Convert leaf 0 into an internal with the two new children.
    h.is_leaf[new_ids[0]] = True
    h.is_leaf[new_ids[1]] = True
    h.convert_leaf_to_internal(0, int(new_ids[0]), int(new_ids[1]))
    assert not bool(h.is_leaf[0])
    assert bool(h.is_leaf[new_ids[0]])
    assert int(h.parent[new_ids[0]]) == 0
