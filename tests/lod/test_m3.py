# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from gsplat.lod import (
    CacheEntry,
    CpuGaussianStore,
    GpuSptCache,
    KnnViewSampler,
    SkyboxSet,
)


def _splats(n: int, K_sh: int = 3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return {
        "means": torch.randn(n, 3, generator=g),
        "scales": torch.rand(n, 3, generator=g),
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "opacities": torch.randn(n, generator=g),
        "sh0": torch.randn(n, 1, 3, generator=g),
        "shN": torch.randn(n, K_sh, 3, generator=g),
    }


# --- CpuGaussianStore -------------------------------------------------------


def test_store_from_splats_roundtrip():
    sp = _splats(10)
    store = CpuGaussianStore.from_splats(sp)
    assert store.N == 10
    assert store.capacity == 10
    out = store.gather(torch.arange(10), device="cpu", copy_adam=True)
    assert torch.allclose(out["means"], sp["means"])
    assert torch.allclose(out["scales"], sp["scales"])
    assert torch.allclose(out["quats"], sp["quats"])
    # Adam moments start at 0.
    assert torch.all(out["exp_avg.means"] == 0)
    assert torch.all(out["step"] == 0)


def test_store_scatter_updates_cpu_state():
    sp = _splats(8)
    store = CpuGaussianStore.from_splats(sp, capacity=16)
    # Pretend GPU returned new means for indices [2, 4, 6].
    idx = torch.tensor([2, 4, 6])
    new_means = torch.ones(3, 3) * 42.0
    store.scatter(idx, {"means": new_means})
    out = store.gather(idx, device="cpu")
    assert torch.allclose(out["means"], new_means)
    # Others unchanged.
    other = store.gather(torch.tensor([0, 1, 3, 5, 7]), device="cpu")
    assert not torch.allclose(other["means"], new_means[0:1].expand(5, 3))


def test_store_allocate_new_grows_capacity():
    sp = _splats(5, K_sh=0)
    store = CpuGaussianStore.from_splats(sp, capacity=5)
    assert store.N == 5
    assert store.capacity == 5
    new = store.allocate_new(10)
    assert new.tolist() == list(range(5, 15))
    assert store.N == 15
    assert store.capacity >= 15


def test_store_scatter_adam_roundtrip():
    sp = _splats(4, K_sh=0)
    store = CpuGaussianStore.from_splats(sp)
    idx = torch.tensor([0, 2])
    adam = {
        "exp_avg.means": torch.ones(2, 3) * 0.5,
        "exp_avg_sq.means": torch.ones(2, 3) * 0.25,
        "step": torch.tensor([10, 20], dtype=torch.int64),
    }
    store.scatter(idx, {}, adam_state=adam)
    out = store.gather(idx, device="cpu", copy_adam=True)
    assert torch.allclose(out["exp_avg.means"], torch.ones(2, 3) * 0.5)
    assert torch.all(out["step"] == torch.tensor([10, 20]))


def test_store_bytes_per_gaussian():
    sp = _splats(1, K_sh=0)
    store = CpuGaussianStore.from_splats(sp)
    # Params: means(3)+scales(3)+quats(4)+opac(1)+sh0(3)+shN(0) = 14 floats.
    # Adam: exp_avg + exp_avg_sq => 28 more floats. Step: 8 bytes (int64).
    # Total: 42*4 + 8 = 176 bytes.
    assert store.bytes_per_gaussian() == 42 * 4 + 8


# --- GpuSptCache ------------------------------------------------------------


def _entry(spt_id: int, size: int, d: float) -> CacheEntry:
    node_ids = torch.arange(size, dtype=torch.long)
    params = {}  # empty: the cache just bookkeeps for these tests
    return CacheEntry(
        spt_id=spt_id, node_ids=node_ids, params=params, cached_distance=d, last_accessed=0
    )


def test_cache_basic_insert_and_lookup():
    c = GpuSptCache(capacity_gaussians=100)
    e = _entry(spt_id=1, size=10, d=5.0)
    c.insert(e)
    hit = c.lookup(1, current_distance=5.0)
    assert hit is e
    assert c.hits == 1 and c.misses == 0


def test_cache_distance_tolerance():
    c = GpuSptCache(capacity_gaussians=100, D_min=0.8, D_max=1.25)
    c.insert(_entry(1, 10, d=10.0))
    # Within tolerance: d_new / d_cached = 9/10 = 0.9, 11/10 = 1.1.
    assert c.lookup(1, current_distance=9.0) is not None
    assert c.lookup(1, current_distance=11.0) is not None
    # Outside tolerance: 7/10 = 0.7 -> miss; 13/10 = 1.3 -> miss.
    assert c.lookup(1, current_distance=7.0) is None
    assert c.lookup(1, current_distance=13.0) is None


def test_cache_lru_eviction_order():
    c = GpuSptCache(capacity_gaussians=30)  # fits 3 entries of size 10
    for i in range(3):
        c.insert(_entry(i, 10, d=5.0))
    # Touch 0 and 2 to bump them up the LRU side.
    c.lookup(0, 5.0)
    c.lookup(2, 5.0)
    # Insert a new entry — must evict the LRU, which is 1.
    c.insert(_entry(3, 10, d=5.0))
    assert 1 not in c
    assert {0, 2, 3} == set(c.keys())


def test_cache_writeback_on_dirty_eviction():
    c = GpuSptCache(capacity_gaussians=20)
    c.insert(_entry(0, 10, d=5.0))
    c.mark_dirty(0)
    written = []
    c.insert(_entry(1, 10, d=5.0), writeback_cb=lambda e: written.append(e.spt_id))
    c.insert(_entry(2, 10, d=5.0), writeback_cb=lambda e: written.append(e.spt_id))
    # 0 is dirty, was evicted (LRU) -> expected in the writeback list.
    assert 0 in written


def test_cache_flush_all_clears():
    c = GpuSptCache(capacity_gaussians=100)
    for i in range(5):
        c.insert(_entry(i, 10, d=5.0))
    c.mark_dirty(2)
    flushed = []
    c.flush_all(writeback_cb=lambda e: flushed.append(e.spt_id))
    assert c.n_entries == 0 and c.total_size == 0
    assert 2 in flushed


# --- KnnViewSampler ---------------------------------------------------------


def test_knn_sampler_returns_valid_views():
    pos = torch.randn(20, 3)
    s = KnnViewSampler(pos, k=5, W=1.0, uniform_every=0, seed=123)
    cur = 0
    for step in range(1, 50):
        cur = s.sample_next(cur, step)
        assert 0 <= cur < 20


def test_knn_sampler_uniform_every_triggers():
    pos = torch.randn(100, 3)
    s = KnnViewSampler(pos, k=3, W=1.0, uniform_every=5, seed=0)
    # Step 5 is uniform; force a deterministic check by running many times
    # and ensuring the sampled view is not always in the neighbourhood of 0.
    cur = 0
    # Count how often we escape the immediate neighbour set of 0 at uniform steps.
    neighbours_of_0 = set(s.neighbours(0).tolist())
    escapes = 0
    for _ in range(200):
        j = s.sample_next(0, 5)
        if j not in neighbours_of_0 and j != 0:
            escapes += 1
    assert escapes > 0  # uniform fallback should sometimes pick far views


def test_knn_sampler_probability_inverse_with_distance():
    # Construct colinear positions where neighbour[0] should be closest (idx 1),
    # then 2, 3, 4. probability for 1 > 2 > 3 > 4.
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0]])
    s = KnnViewSampler(pos, k=5, W=0.1, uniform_every=0, seed=0)
    probs = s.distribution(0)
    nn = s.neighbours(0)
    # Zip and sort by distance from 0 to check prob monotonicity.
    order = np.argsort(np.abs(nn - 0))  # sort by index (==distance here)
    sorted_probs = probs[order]
    assert all(sorted_probs[i] >= sorted_probs[i + 1] for i in range(len(sorted_probs) - 1))


# --- SkyboxSet --------------------------------------------------------------


def test_skybox_points_on_sphere():
    centre = torch.zeros(3)
    sk = SkyboxSet.make_icosphere(n_points=200, radius=10.0, centre=centre, sh_degree=0, device="cpu")
    r = sk.means.norm(dim=-1)
    assert torch.allclose(r, torch.full_like(r, 10.0), atol=1e-4)
    assert sk.shN.shape == (200, 0, 3)
    assert sk.sh0.shape == (200, 1, 3)
