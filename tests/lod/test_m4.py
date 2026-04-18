# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from gsplat.lod import (
    CpuGaussianStore,
    HierarchicalMcmcDensifier,
    OutOfCoreAdam,
    build_hierarchy,
    build_hspt,
)


def _splats(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return {
        "means": torch.randn(n, 3, generator=g),
        "scales": (torch.rand(n, 3, generator=g) * 0.5 - 2.5),
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "opacities": torch.full((n,), 2.0),
        "sh0": torch.randn(n, 1, 3, generator=g) * 0.3,
        "shN": torch.zeros(n, 0, 3),
    }


# --- OutOfCoreAdam ----------------------------------------------------------


def test_out_of_core_adam_equivalent_to_torch_adam_on_full_set():
    n = 6
    sp = _splats(n)
    store = CpuGaussianStore.from_splats(sp)

    # Reference: a plain torch Adam.
    p_ref = torch.nn.Parameter(store.means.clone())
    opt_ref = torch.optim.Adam([p_ref], lr=1e-2, betas=(0.9, 0.999), eps=1e-8)

    oc = OutOfCoreAdam(store, lr_spec={"means": 1e-2})

    for step in range(5):
        # Same gradient applied to both.
        grad = torch.randn(n, 3, generator=torch.Generator().manual_seed(step))
        # Reference.
        opt_ref.zero_grad()
        p_ref.grad = grad.clone()
        opt_ref.step()

        # OutOfCore.
        active = torch.arange(n, dtype=torch.long)
        active_p = torch.nn.Parameter(store.means[active].clone().cpu())
        active_p.grad = grad.clone()
        oc.step(active_indices=active, active_params={"means": active_p})

    # After N identical updates, the two sets of means should be close.
    assert torch.allclose(store.means, p_ref.data, atol=1e-5), (
        f"max diff: {(store.means - p_ref.data).abs().max().item()}"
    )


def test_out_of_core_adam_sparse_subset_leaves_others_untouched():
    n = 5
    sp = _splats(n, seed=2)
    store = CpuGaussianStore.from_splats(sp)
    # Update only indices 0 and 3.
    active = torch.tensor([0, 3], dtype=torch.long)
    oc = OutOfCoreAdam(store, lr_spec={"means": 1e-2})
    means0 = store.means.clone()
    g = torch.ones(2, 3)
    p = torch.nn.Parameter(store.means[active].clone().cpu())
    p.grad = g
    oc.step(active_indices=active, active_params={"means": p})
    # Active ones should have moved.
    assert not torch.allclose(store.means[active], means0[active])
    # Inactive ones should be unchanged.
    others = torch.tensor([1, 2, 4])
    assert torch.allclose(store.means[others], means0[others])
    # Their Adam step counter should still be 0.
    assert torch.all(store.step[others] == 0)
    assert torch.all(store.step[active] == 1)


# --- HierarchicalMcmcDensifier ---------------------------------------------


def _densifier_with(n: int = 8, seed: int = 10):
    sp = _splats(n, seed=seed)
    h = build_hierarchy(sp)
    store = CpuGaussianStore.from_splats(sp, capacity=n * 4)
    # After build_hierarchy, internals live at indices [n, 2n-1). Extend the
    # store to cover them too so densifier can maintain the invariant.
    store.N = h.n_total
    # Copy all hierarchy node params into the store so store[i] == hierarchy[i].
    store.means[: h.n_total] = h.mu
    store.scales[: h.n_total] = h.scale
    store.quats[: h.n_total] = h.quat
    store.opacities[: h.n_total] = h.opacity
    store.sh0[: h.n_total] = h.sh0
    store.shN[: h.n_total] = h.shN
    d = HierarchicalMcmcDensifier(
        hierarchy=h,
        store=store,
        hspt_size_threshold=1e-3,
    )
    return d, h, store


def test_spawn_grows_tree_by_two_per_candidate():
    d, h, store = _densifier_with(n=8, seed=1)
    # Pick one live leaf to spawn.
    leaf_ids = torch.nonzero(h.is_leaf, as_tuple=False).squeeze(-1)
    c = int(leaf_ids[0].item())
    old_total = h.n_total
    old_store_N = store.N
    d._spawn(c)
    # c should now be internal; two new leaves appended.
    assert not bool(h.is_leaf[c])
    assert h.n_total == old_total + 2
    assert store.N == old_store_N + 2
    # The two new indices should be the final two in the hierarchy/store.
    new_l = int(h.left[c].item())
    new_r = int(h.right[c].item())
    assert new_l == old_total and new_r == old_total + 1
    # Children are leaves.
    assert bool(h.is_leaf[new_l]) and bool(h.is_leaf[new_r])
    # Tree still valid.
    h.validate()


def test_respawn_keeps_tree_valid():
    d, h, store = _densifier_with(n=10, seed=2)
    # Pick any pair: dead leaf + live target leaf.
    leaf_ids = torch.nonzero(h.is_leaf, as_tuple=False).squeeze(-1).tolist()
    assert len(leaf_ids) >= 3
    dead = leaf_ids[0]
    target = leaf_ids[-1]
    d._respawn(dead, target)
    h.validate()
    # After respawn, target must be internal; dead must still be a leaf but
    # reparented under target.
    assert not bool(h.is_leaf[target])
    assert bool(h.is_leaf[dead])
    assert int(h.parent[dead]) == target


def test_densify_step_invariants():
    d, h, store = _densifier_with(n=16, seed=3)
    # Simulate a few iterations of grad observation: random non-zero grads.
    d.grad_stats.resize(store.N)
    leaf_ids = torch.nonzero(h.is_leaf, as_tuple=False).squeeze(-1)
    d.grad_stats.observe(leaf_ids, torch.rand(leaf_ids.numel()) + 0.01, step=5)

    old_leaves = int(h.is_leaf.sum().item())
    hspt = d.densify_step(step=100, n_spawn=3, n_respawn=0)
    h.validate()
    # We added 3 spawn pairs ⇒ +3 internals (old leaves promoted), +6 leaves.
    # Net leaves: old + 6 - 3 (promoted) = old + 3.
    new_leaves = int(h.is_leaf.sum().item())
    assert new_leaves == old_leaves + 3
    assert hspt.n_spts > 0
