# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Out-of-core LoD trainer for gsplat.

Implements the two-stage pipeline from "A LoD of Gaussians" (Windisch et al.,
2026, arXiv:2507.01110v4):

    Stage 1 (coarse): standard 3DGS-MCMC training on a GPU-resident splat up
                      to ``cfg.coarse_iters`` iterations.

    Stage 2 (LoD):    build a ``GaussianHierarchy`` from the trained leaves;
                      move all per-Gaussian state to a ``CpuGaussianStore``;
                      build an HSPT. Each subsequent iteration samples a
                      view, computes a render set via the cut, pulls the
                      subset to GPU, renders, and updates Adam in place on
                      the CPU store. ``LoDMCMCStrategy`` drives densification
                      / respawning as the hierarchy evolves.

This is a lean end-to-end script; ``examples/simple_trainer.py`` remains the
full-featured reference trainer. For production work, wire up schedulers,
logging, and checkpointing as needed.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gsplat.rendering import rasterization
from gsplat.strategy import LoDMCMCStrategy, MCMCStrategy

from gsplat.lod import (
    CpuGaussianStore,
    GpuSptCache,
    CacheEntry,
    HierarchicalMcmcDensifier,
    KnnViewSampler,
    LoDConfig,
    OutOfCoreAdam,
    SkyboxSet,
    build_hierarchy,
    build_hspt,
    compute_render_set,
)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    # Dataset ---------------------------------------------------------------
    data_dir: str = "data/360_v2/garden"
    data_factor: int = 4
    test_every: int = 8
    result_dir: str = "results/lod_garden"

    # Stage 1 (coarse) ------------------------------------------------------
    coarse_iters: int = 100_000
    coarse_cap_max: int = 5_000_000
    coarse_refine_start: int = 500
    coarse_refine_stop: int = 25_000
    coarse_refine_every: int = 100
    coarse_init_num_pts: int = 100_000
    coarse_init_extent: float = 3.0
    coarse_init_opacity: float = 0.1
    coarse_init_scale: float = 1.0

    # Stage 2 (LoD) ---------------------------------------------------------
    lod: LoDConfig = field(default_factory=LoDConfig)

    # Learning rates --------------------------------------------------------
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20.0

    # SH degree schedule ----------------------------------------------------
    sh_degree: int = 3

    # Runtime ---------------------------------------------------------------
    device: str = "cuda"
    seed: int = 0
    skip_stage1: bool = False  # load coarse splats from disk instead
    coarse_ckpt: Optional[str] = None

    # Logging ---------------------------------------------------------------
    log_every: int = 100
    save_every: int = 10_000


# -----------------------------------------------------------------------------
# Stage 1 helpers (minimal coarse trainer — mirrors simple_trainer's MCMC path)
# -----------------------------------------------------------------------------


def _l1_loss(pred: Tensor, gt: Tensor) -> Tensor:
    return (pred - gt).abs().mean()


def _rgb_to_sh0(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def _random_init_splats(
    parser,
    cfg: TrainerConfig,
    device: str,
) -> Dict[str, nn.Parameter]:
    """Either initialise from the COLMAP point cloud or fall back to random."""
    N = int(cfg.coarse_init_num_pts)
    sh_degree = cfg.sh_degree

    if parser.points is not None and parser.points.shape[0] > 0:
        pts = torch.from_numpy(parser.points).float()
        cols = torch.from_numpy(parser.points_rgb).float() / 255.0
        if pts.shape[0] > N:
            idx = torch.randperm(pts.shape[0])[:N]
            pts = pts[idx]
            cols = cols[idx]
        N = pts.shape[0]
    else:
        extent = cfg.coarse_init_extent
        pts = (torch.rand(N, 3) - 0.5) * 2.0 * extent
        cols = torch.rand(N, 3)

    # Per-Gaussian isotropic scale based on 3 nearest-neighbour distance.
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts.numpy())
        dists = tree.query(pts.numpy(), k=4)[0][:, 1:].mean(axis=1)
        scales = torch.from_numpy(np.log(np.maximum(dists, 1e-7))).float()
    except Exception:
        scales = torch.full((N,), math.log(0.01))
    scales = scales.unsqueeze(-1).expand(N, 3).contiguous()

    quats = torch.zeros(N, 4)
    quats[:, 0] = 1.0

    opacities = torch.logit(torch.full((N,), float(cfg.coarse_init_opacity)))

    K = (sh_degree + 1) ** 2 - 1
    sh0 = _rgb_to_sh0(cols).view(N, 1, 3)
    shN = torch.zeros(N, K, 3)

    params = {
        "means": nn.Parameter(pts.to(device)),
        "scales": nn.Parameter(scales.to(device)),
        "quats": nn.Parameter(quats.to(device)),
        "opacities": nn.Parameter(opacities.to(device)),
        "sh0": nn.Parameter(sh0.to(device)),
        "shN": nn.Parameter(shN.to(device)),
    }
    return params


def _build_coarse_optimizers(
    params: Dict[str, nn.Parameter], cfg: TrainerConfig
) -> Dict[str, torch.optim.Optimizer]:
    lr_map = {
        "means": cfg.lr_means,
        "scales": cfg.lr_scales,
        "quats": cfg.lr_quats,
        "opacities": cfg.lr_opacities,
        "sh0": cfg.lr_sh0,
        "shN": cfg.lr_shN,
    }
    opts = {}
    for k, p in params.items():
        opts[k] = torch.optim.Adam([p], lr=lr_map[k], eps=1e-15)
    return opts


def _stage1_coarse_train(
    cfg: TrainerConfig,
    parser,
    trainset,
    device: str,
) -> Dict[str, Tensor]:
    """Run the coarse MCMC stage and return the splat dict on CPU."""
    import tqdm

    print(f"[stage1] coarse MCMC training for {cfg.coarse_iters} iterations")
    torch.manual_seed(cfg.seed)
    params = _random_init_splats(parser, cfg, device)
    optimizers = _build_coarse_optimizers(params, cfg)

    strategy = MCMCStrategy(
        cap_max=cfg.coarse_cap_max,
        noise_lr=5e5,
        refine_start_iter=cfg.coarse_refine_start,
        refine_stop_iter=cfg.coarse_refine_stop,
        refine_every=cfg.coarse_refine_every,
        min_opacity=0.005,
    )
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()

    pbar = tqdm.tqdm(range(cfg.coarse_iters), desc="stage1 coarse")
    n_train = len(trainset)
    for step in pbar:
        data = trainset[int(torch.randint(0, n_train, (1,)).item())]
        gt = (data["image"] / 255.0).to(device).unsqueeze(0)
        viewmat = torch.linalg.inv(data["camtoworld"].to(device)).unsqueeze(0)
        K = data["K"].to(device).unsqueeze(0)
        H, W = int(gt.shape[1]), int(gt.shape[2])

        strategy.step_pre_backward(params, optimizers, state, step, info={})
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        render, _, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=params["scales"],
            opacities=params["opacities"],
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            sh_degree=cfg.sh_degree,
            packed=False,
        )
        loss = _l1_loss(render, gt)
        loss.backward()

        strategy.step_post_backward(
            params, optimizers, state, step, info, lr=cfg.lr_means
        )
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        if step % cfg.log_every == 0:
            pbar.set_postfix(
                n=int(params["means"].shape[0]), loss=float(loss.item())
            )

    # Move everything to CPU for hierarchy construction.
    out = {k: p.detach().cpu() for k, p in params.items()}
    return out


# -----------------------------------------------------------------------------
# Stage 2 (LoD) pipeline
# -----------------------------------------------------------------------------


def _writeback_cb_factory(store: CpuGaussianStore):
    def cb(entry: CacheEntry) -> None:
        if not entry.params:
            return
        store.scatter(
            entry.node_ids.to("cpu"),
            {k: v for k, v in entry.params.items() if k in ("means", "scales", "quats", "opacities", "sh0", "shN")},
        )

    return cb


def _active_params_from_store(
    store: CpuGaussianStore, indices: Tensor, device: str
) -> Dict[str, nn.Parameter]:
    """Pull an active subset from the CPU store to GPU as leaf Parameters."""
    bag = store.gather(indices, device=device)
    return {
        k: nn.Parameter(bag[k].contiguous(), requires_grad=True)
        for k in ("means", "scales", "quats", "opacities", "sh0", "shN")
    }


def _stage2_lod_train(
    cfg: TrainerConfig,
    parser,
    trainset,
    coarse_splats: Dict[str, Tensor],
    device: str,
) -> None:
    import tqdm

    print(f"[stage2] building hierarchy from {coarse_splats['means'].shape[0]} leaves")
    t0 = time.time()
    hierarchy = build_hierarchy(coarse_splats)
    print(f"[stage2] hierarchy built in {time.time() - t0:.1f}s: {hierarchy.n_total} nodes, {hierarchy.n_leaves} leaves")

    # CPU store covers all hierarchy nodes (not just leaves).
    store = CpuGaussianStore.from_splats(
        {
            "means": hierarchy.mu,
            "scales": hierarchy.scale,
            "quats": hierarchy.quat,
            "opacities": hierarchy.opacity,
            "sh0": hierarchy.sh0,
            "shN": hierarchy.shN,
        },
        capacity=int(hierarchy.n_total * 2),
    )

    hspt = build_hspt(hierarchy, size=cfg.lod.hspt_size_threshold, refined_m_d=cfg.lod.use_refined_m_d)
    print(f"[stage2] HSPT: {hspt.n_upper} upper nodes, {hspt.n_spts} SPTs, {hspt.n_entries} entries")

    cache = GpuSptCache(
        capacity_gaussians=cfg.lod.cache_capacity_gaussians,
        D_min=cfg.lod.cache_D_min,
        D_max=cfg.lod.cache_D_max,
        device=device,
    )

    # View positions: camera centres from COLMAP parser's camtoworlds.
    cam_centres = torch.from_numpy(parser.camtoworlds[:, :3, 3]).float()
    sampler = KnnViewSampler(
        cam_centres,
        k=cfg.lod.knn_k,
        W=cfg.lod.knn_W,
        uniform_every=cfg.lod.uniform_view_every,
        seed=cfg.seed,
    )

    densifier = HierarchicalMcmcDensifier(
        hierarchy=hierarchy,
        store=store,
        hspt_size_threshold=cfg.lod.hspt_size_threshold,
        respawn_opacity_threshold=cfg.lod.respawn_opacity_threshold,
        respawn_unused_iters=cfg.lod.respawn_unused_iters,
        refined_m_d=cfg.lod.use_refined_m_d,
    )
    strategy = LoDMCMCStrategy(cfg=cfg.lod)

    oc_optim = OutOfCoreAdam(
        store,
        lr_spec={
            "means": cfg.lr_means,
            "scales": cfg.lr_scales,
            "quats": cfg.lr_quats,
            "opacities": cfg.lr_opacities,
            "sh0": cfg.lr_sh0,
            "shN": cfg.lr_shN,
        },
    )

    # Optional skybox, kept GPU-resident for every render.
    skybox = None
    if cfg.lod.skybox_enabled:
        scene_centre = torch.from_numpy(parser.points.mean(0)).float() if parser.points is not None else torch.zeros(3)
        skybox = SkyboxSet.make_icosphere(
            n_points=cfg.lod.skybox_n_points,
            radius=cfg.lod.skybox_radius,
            centre=scene_centre,
            sh_degree=cfg.sh_degree,
            device=device,
        )

    state: Dict[str, object] = {
        "densifier": densifier,
        "cache": cache,
        "hspt": hspt,
        "store": store,
        "writeback_cb": _writeback_cb_factory(store),
    }

    view_idx = 0
    pbar = tqdm.tqdm(range(cfg.lod.lod_iters), desc="stage2 LoD")
    for step in pbar:
        hspt = state["hspt"]  # may have been rebuilt by strategy
        view_idx = sampler.sample_next(view_idx, step)
        data = trainset[view_idx % len(trainset)]
        gt = (data["image"] / 255.0).to(device).unsqueeze(0)
        camtoworld = data["camtoworld"].to(device)
        viewmat = torch.linalg.inv(camtoworld).unsqueeze(0)
        K = data["K"].to(device).unsqueeze(0)
        H, W = int(gt.shape[1]), int(gt.shape[2])
        camera_pos = camtoworld[:3, 3]

        # Render-set cut.
        rs = compute_render_set(
            hspt,
            camera_pos.detach().cpu(),
            viewmat[0].detach().cpu(),
            K[0].detach().cpu(),
            W,
            H,
            T=cfg.lod.T,
            frustum_radius_mult=cfg.lod.frustum_radius_mult,
            refined_m_d=cfg.lod.use_refined_m_d,
        )
        if rs.node_ids.numel() == 0 and skybox is None:
            continue

        # Pull active subset to GPU (params are ephemeral Parameters).
        active_indices = rs.node_ids  # these ARE CpuGaussianStore indices
        active = _active_params_from_store(store, active_indices, device)

        # Concatenate with skybox for rendering only (skybox isn't optimised by
        # the LoD subsystem — keep it trainable via a separate optimiser if
        # desired).
        if skybox is not None:
            means = torch.cat([active["means"], skybox.means], dim=0)
            scales = torch.cat([active["scales"], skybox.scales], dim=0)
            quats = torch.cat([active["quats"], skybox.quats], dim=0)
            opacities = torch.cat([active["opacities"], skybox.opacities], dim=0)
            sh0 = torch.cat([active["sh0"], skybox.sh0], dim=0)
            shN = torch.cat([active["shN"], skybox.shN], dim=0)
        else:
            means, scales, quats = active["means"], active["scales"], active["quats"]
            opacities, sh0, shN = active["opacities"], active["sh0"], active["shN"]

        colors = torch.cat([sh0, shN], dim=1)

        # Strategy pre-backward hook.
        state["active_indices"] = active_indices
        strategy.step_pre_backward(active, oc_optim, state, step, info={})

        render, _, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            sh_degree=cfg.sh_degree,
            packed=False,
        )

        # Loss and backward.
        loss = _l1_loss(render, gt)
        loss.backward()

        # Strategy post-backward (densify + noise + cache flush).
        strategy.step_post_backward(active, oc_optim, state, step, info, lr=cfg.lr_means)

        # OutOfCoreAdam step on the active subset.
        oc_optim.step(active_indices=active_indices, active_params=active)

        # Populate cache with the touched SPTs (best-effort reuse).
        for spt_id, d_spt in zip(rs.spt_ids_touched.tolist(), rs.spt_distances.tolist()):
            if spt_id in cache:
                cache.mark_dirty(spt_id)

        if step % cfg.log_every == 0:
            pbar.set_postfix(
                n=int(active_indices.numel()),
                ncache=len(cache),
                loss=float(loss.item()),
            )

    # Persist final store.
    out_dir = Path(cfg.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "means": store.means[: store.N].clone(),
        "scales": store.scales[: store.N].clone(),
        "quats": store.quats[: store.N].clone(),
        "opacities": store.opacities[: store.N].clone(),
        "sh0": store.sh0[: store.N].clone(),
        "shN": store.shN[: store.N].clone(),
    }
    torch.save(ckpt, out_dir / "lod_final.pt")
    print(f"[stage2] saved {out_dir / 'lod_final.pt'}  ({store.N} Gaussians)")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main(cfg: TrainerConfig) -> None:
    # Defer these imports so that the trainer module can be imported without
    # the dataset dependencies (cv2, imageio, ...) being installed.
    import sys as _sys

    _examples_dir = os.path.dirname(os.path.abspath(__file__))
    if _examples_dir not in _sys.path:
        _sys.path.insert(0, _examples_dir)
    from datasets.colmap import Dataset, Parser  # type: ignore

    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    if device != cfg.device:
        print(f"[warn] requested device {cfg.device!r} unavailable; using {device!r}")

    Path(cfg.result_dir).mkdir(parents=True, exist_ok=True)

    parser = Parser(
        data_dir=cfg.data_dir,
        factor=cfg.data_factor,
        normalize=True,
        test_every=cfg.test_every,
    )
    trainset = Dataset(parser, split="train")

    if cfg.skip_stage1:
        assert cfg.coarse_ckpt is not None, "skip_stage1 requires coarse_ckpt"
        coarse_splats = torch.load(cfg.coarse_ckpt, map_location="cpu")
    else:
        coarse_splats = _stage1_coarse_train(cfg, parser, trainset, device)
        coarse_ckpt_path = Path(cfg.result_dir) / "coarse.pt"
        torch.save(coarse_splats, coarse_ckpt_path)
        print(f"[stage1] saved coarse checkpoint to {coarse_ckpt_path}")

    _stage2_lod_train(cfg, parser, trainset, coarse_splats, device)


if __name__ == "__main__":
    import tyro

    main(tyro.cli(TrainerConfig))
