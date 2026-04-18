# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration test for M5: stage-2 LoD loop plumbing.

Skips the rasterization call (which needs CUDA) and instead synthesises
gradients, but wires every other component together:
    HSPT cut -> CpuGaussianStore.gather -> LoDMCMCStrategy -> OutOfCoreAdam
    -> densification -> HSPT rebuild -> next iter.

The assertion is that the loop runs without exceptions, the CPU store grows
monotonically when densification is enabled, and Adam state / step counters
are updated exactly for active indices.
"""

import torch

from gsplat.lod import (
    CpuGaussianStore,
    GpuSptCache,
    HierarchicalMcmcDensifier,
    KnnViewSampler,
    LoDConfig,
    OutOfCoreAdam,
    build_hierarchy,
    build_hspt,
    compute_render_set,
)
from gsplat.strategy import LoDMCMCStrategy


def _splats(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return {
        "means": torch.randn(n, 3, generator=g) * 1.5 + torch.tensor([0.0, 0.0, 5.0]),
        "scales": (torch.rand(n, 3, generator=g) * 0.3 - 2.5),
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "opacities": torch.full((n,), 2.0),
        "sh0": torch.zeros(n, 1, 3),
        "shN": torch.zeros(n, 0, 3),
    }


def _setup(n_init: int = 64, seed: int = 0):
    sp = _splats(n_init, seed=seed)
    h = build_hierarchy(sp)
    store = CpuGaussianStore.from_splats(
        {
            "means": h.mu,
            "scales": h.scale,
            "quats": h.quat,
            "opacities": h.opacity,
            "sh0": h.sh0,
            "shN": h.shN,
        },
        capacity=h.n_total * 4,
    )
    hspt = build_hspt(h, size=1e-3)
    cfg = LoDConfig()
    cfg.lod_iters = 30
    cfg.densify_start_iter = 3
    cfg.densify_every = 5
    cfg.densify_stop_iter = 25
    cfg.lod_cap_max = h.n_total * 3
    cache = GpuSptCache(capacity_gaussians=10_000, device="cpu")
    densifier = HierarchicalMcmcDensifier(
        hierarchy=h, store=store, hspt_size_threshold=1e-3
    )
    strategy = LoDMCMCStrategy(cfg=cfg)
    oc_optim = OutOfCoreAdam(
        store,
        lr_spec={"means": 1e-3, "scales": 1e-3, "quats": 1e-3,
                 "opacities": 1e-2, "sh0": 1e-3, "shN": 1e-3},
    )
    return cfg, store, h, hspt, cache, densifier, strategy, oc_optim


def test_stage2_loop_runs_and_grows():
    cfg, store, h, hspt, cache, densifier, strategy, oc_optim = _setup()

    state = {
        "densifier": densifier,
        "cache": cache,
        "hspt": hspt,
        "store": store,
    }

    # Pre-seed grad stats across all leaves so spawn has candidates.
    leaf_ids = torch.nonzero(h.is_leaf, as_tuple=False).squeeze(-1)
    densifier.grad_stats.observe(leaf_ids, torch.rand(leaf_ids.numel()) + 0.01, step=0)

    initial_N = store.N
    viewmat = torch.eye(4)
    K = torch.tensor([[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]])

    for step in range(cfg.densify_start_iter + 2 * cfg.densify_every):
        rs = compute_render_set(
            state["hspt"], torch.zeros(3), viewmat, K, 200, 200, T=cfg.T
        )
        state["active_indices"] = rs.node_ids
        # Re-seed grad stats each step so densify always has candidates.
        densifier.grad_stats.observe(
            leaf_ids, torch.rand(leaf_ids.numel()) + 0.01, step=step
        )
        active = {
            k: torch.nn.Parameter(store._param_tensor(k)[rs.node_ids].clone())
            for k in ("means", "scales", "quats", "opacities", "sh0", "shN")
        }
        for k in active:
            active[k].grad = torch.randn_like(active[k].data) * 1e-3

        strategy.step_post_backward(active, oc_optim, state, step, info={}, lr=1e-4)
        oc_optim.step(rs.node_ids, active)

    # After the loop, the store must have grown via densification.
    assert store.N > initial_N, (
        f"expected densification to grow store beyond {initial_N}; got {store.N}"
    )


def test_stage2_loop_adam_updates_active_subset_only():
    cfg, store, h, hspt, cache, densifier, strategy, oc_optim = _setup(n_init=8)
    cfg.densify_start_iter = 10_000  # disable densification for this test

    state = {
        "densifier": densifier,
        "cache": cache,
        "hspt": hspt,
        "store": store,
    }

    # Pick a sparse active subset.
    active_ids = torch.tensor([0, 2, 4], dtype=torch.long)
    means_before = store.means.clone()
    step_before = store.step.clone()

    active = {
        k: torch.nn.Parameter(store._param_tensor(k)[active_ids].clone())
        for k in ("means", "scales", "quats", "opacities", "sh0", "shN")
    }
    for k in active:
        active[k].grad = torch.ones_like(active[k].data) * 0.5
    state["active_indices"] = active_ids

    strategy.step_post_backward(active, oc_optim, state, step=0, info={}, lr=1e-4)
    oc_optim.step(active_ids, active)

    # Updates only at active_ids.
    for i in active_ids.tolist():
        assert not torch.allclose(store.means[i], means_before[i])
        assert store.step[i] == step_before[i] + 1
    # Untouched elsewhere.
    inactive = torch.tensor([1, 3, 5, 6, 7])
    assert torch.allclose(store.means[inactive], means_before[inactive])
    assert torch.all(store.step[inactive] == step_before[inactive])
