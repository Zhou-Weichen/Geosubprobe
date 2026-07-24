# Copyright (c) 2026 Weichen Zhou.
# Released under the MIT License (see the LICENSE file at the repository root).
#
# This file implements the Subspace Intervention method proposed in our paper.
# It builds on infrastructure from Probe3D (https://github.com/mbanani/probe3d),
# Copyright (c) 2024 Mohamed El Banani, also released under the MIT License.

from __future__ import annotations

import argparse
import os
import re
import shlex
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from torch.nn.functional import interpolate
from tqdm import tqdm

from evals.datasets.builder import build_loader
from evals.utils.losses import DepthLoss
from evals.utils.metrics import evaluate_depth

# Maps the paper's named interventions onto the two internal knobs
# (mode = how V_k is built, part_mode = which part of the feature is kept).
INTERVENTIONS = {
    "aligned": dict(mode="svd", part_mode="proj"),      # Eq. 3
    "residual": dict(mode="svd", part_mode="res"),      # Eq. 4
    "random": dict(mode="random", part_mode="proj"),    # R_k null baseline
    "shifted": dict(mode="shifted", part_mode="proj"),  # non-top-k ablation
    "full": dict(mode=None, part_mode="full"),          # full-rank upper bound
}

# state_dict key of a linear (1x1 conv) probe head
LINEAR_WEIGHT_KEY = "head.conv.weight"


def visualize_depth_map(
    depth_tensor,
    cmap_name="viridis",
    depth_min=0,
    depth_max=10.0,
    inverse=False,
    percentile_clipping=True,
    low_p=1,
    high_p=99,
    gamma=0.8,
):
    import matplotlib.pyplot as plt

    if depth_tensor.ndim == 4:
        depth = depth_tensor[0, 0].cpu().numpy()
    elif depth_tensor.ndim == 2:
        depth = depth_tensor.cpu().numpy()
    else:
        depth = depth_tensor[0, 0].cpu().numpy()

    if percentile_clipping:
        valid_depth = depth[depth > depth_min]
        p_min = np.percentile(valid_depth, low_p) if valid_depth.size > 0 else depth_min
        p_max = np.percentile(valid_depth, high_p) if valid_depth.size > 0 else depth_max
        clipped_depth = np.clip(depth, p_min, p_max)
    else:
        clipped_depth = np.clip(depth, depth_min, depth_max)

    norm_min = np.min(clipped_depth)
    norm_max = np.max(clipped_depth)
    if norm_max - norm_min < 1e-6:
        depth_norm = np.zeros_like(clipped_depth)
    else:
        depth_norm = (clipped_depth - norm_min) / (norm_max - norm_min)

    if gamma != 1:
        depth_norm = depth_norm ** gamma
    if inverse:
        depth_norm = 1.0 - depth_norm

    cmap = plt.colormaps[cmap_name]
    colored = cmap(depth_norm)[..., :3]
    return (colored * 255).astype(np.uint8)


def compute_subspace(W_mat, k, mode, shift=0):
    """
    Args:
        W_mat: probe weight reshaped to [C, D] (C output channels, D feature dim).
        k:     target subspace rank.
        mode:  "svd"     (top-k right singular vectors), 
               "random"  (k random orthonormal directions in R^D),
               "shifted" (right singular vectors [shift:shift+k]).
        shift: starting index for "shifted" mode.

    Returns:
        V_k: [D, k] tensor with orthonormal columns.
    """
    C, D = W_mat.shape
    # rank is upper-bounded by C (C << D); guard against over-requesting.
    if k > C:
        logger.warning(f"Requested k={k} exceeds rank bound C={C}; clamping to {C}.")
        k = C

    if mode == "svd":
        _, _, Vh = torch.linalg.svd(W_mat, full_matrices=False)  # Vh: [C, D]
        V_k = Vh[:k].T.contiguous()  # [D, k]
    elif mode == "random":
        R = torch.randn(D, k, device=W_mat.device, dtype=W_mat.dtype)
        Q, _ = torch.linalg.qr(R)
        V_k = Q[:, :k]
    elif mode == "shifted":
        _, _, Vh = torch.linalg.svd(W_mat, full_matrices=False)
        assert shift + k <= Vh.shape[0], \
            f"shift+k={shift + k} exceeds rank {Vh.shape[0]}"
        V_k = Vh[shift:shift + k].T.contiguous()
    else:
        raise ValueError(f"Unknown mode '{mode}'. Supported: ['svd', 'random', 'shifted'].")

    return V_k.contiguous()


def sync_backbone_layer(cfg, ckpt, overrides):
    # Recover the feature-extraction layer from the checkpoint's training config.

    if not (isinstance(ckpt, dict) and ckpt.get("cfg") is not None):
        return
    trained_layer = OmegaConf.select(OmegaConf.create(ckpt["cfg"]), "backbone.layer")
    if trained_layer is None:
        return

    user_set_layer = any(re.match(r"\+*backbone\.layer=", o) for o in overrides)
    current_layer = OmegaConf.select(cfg, "backbone.layer")
    if not user_set_layer:
        OmegaConf.update(cfg, "backbone.layer", trained_layer, force_add=True)
        logger.info(
            f"Recovered backbone.layer={trained_layer} from checkpoint "
            f"(the probe was trained on this layer)."
        )
    elif current_layer != trained_layer:
        logger.warning(
            f"backbone.layer={current_layer} was passed via --overrides, but the probe "
            f"was trained on layer {trained_layer}; extracted features will NOT match "
            f"training and metrics will be meaningless."
        )


def project_geo(f, V_k):
    # Orthogonal projection onto the subspace spanned by V_k: Z V_k V_k^T.
    B, D, H, W = f.shape
    f_flat = f.permute(0, 2, 3, 1).reshape(-1, D)  # [BHW, D]
    z = f_flat @ V_k                                # [BHW, k]
    f_proj = z @ V_k.T                              # [BHW, D]
    return f_proj.view(B, H, W, D).permute(0, 3, 1, 2)


def match_energy_to_ref(feat_part, feat_ref, eps=1e-6):
    """Scale feat_part so its per-sample Frobenius norm matches feat_ref.

    Both are [B, D, H, W]. Used to disentangle "subspace direction" from "energy
    retained" in the intervention.
    """
    ref = feat_ref.flatten(1).norm(p=2, dim=1)   # [B]
    part = feat_part.flatten(1).norm(p=2, dim=1)  # [B]
    scale = (ref / (part + eps)).view(-1, 1, 1, 1)
    return feat_part * scale


def apply_intervention(feat, V_k, part_mode, energy_ctrl):
    """Return the intervened feature that will be fed to the frozen probe."""
    feat_full = torch.cat(feat, dim=1) if isinstance(feat, list) else feat

    if part_mode == "full":
        feat_part = feat_full
    elif part_mode == "proj":
        feat_part = project_geo(feat_full, V_k)          # Z V_k V_k^T   (Eq. 3)
    elif part_mode == "res":
        feat_part = feat_full - project_geo(feat_full, V_k)  # Z - Z V_k V_k^T (Eq. 4)
    else:
        raise ValueError(f"Unknown part_mode: {part_mode}")

    if energy_ctrl == "none":
        return feat_part
    elif energy_ctrl == "match_full":
        return match_energy_to_ref(feat_part, feat_full)
    else:
        raise ValueError(f"Unknown energy_ctrl: {energy_ctrl}")


def validate_depth(
    model,
    probe,
    loader,
    loss_fn,
    V_k,
    layer=None,
    scale_invariant=False,
    save_pred=False,
    verbose=True,
    part_mode="proj",
    energy_ctrl="none",
):
    total_loss = 0.0
    metrics = None

    if save_pred:
        subdir = f"l{layer}" if layer is not None else ""
        output_dir = Path("./Visualization/sgeo/val_depth") / loader.dataset.name / subdir
        target_dir = Path("./Visualization/target") / loader.dataset.name
        output_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        iterator = tqdm(loader, desc="Evaluation") if verbose else loader
        for i, batch in enumerate(iterator):
            images = batch["image"].cuda()
            target = batch["depth"].cuda()

            feat = model(images)

            if V_k is not None or part_mode == "full":
                feat_geo = apply_intervention(feat, V_k, part_mode, energy_ctrl)
                pred = probe(feat_geo)
            else:
                pred = probe(feat)

            pred = interpolate(
                pred, size=target.shape[-2:], mode="bilinear", align_corners=True
            )

            if save_pred:
                Image.fromarray(visualize_depth_map(pred)).save(output_dir / f"{i:06d}.png")
                Image.fromarray(visualize_depth_map(target)).save(target_dir / f"{i:06d}.png")

            loss = loss_fn(pred, target)
            total_loss += loss.item()

            batch_metrics = evaluate_depth(pred, target, scale_invariant=scale_invariant)
            if metrics is None:
                metrics = {k: [v] for k, v in batch_metrics.items()}
            else:
                for k, v in batch_metrics.items():
                    metrics[k].append(v)

    total_loss /= len(loader)
    for k in metrics:
        metrics[k] = torch.cat(metrics[k], dim=0).mean()

    return total_loss, metrics


def evaluate(
    ckpt_path,
    intervention="aligned",
    k=32,
    shift=0,
    energy_ctrl="none",
    save_pred=False,
    layer=None,
    config_path="./configs",
    config_name="depth_training",
    cfg_overrides=None,
):
    os.environ.setdefault("XFORMERS_DISABLED", "1")

    if intervention not in INTERVENTIONS:
        raise ValueError(
            f"Unknown intervention '{intervention}'. Choices: {list(INTERVENTIONS)}"
        )
    mode = INTERVENTIONS[intervention]["mode"]
    part_mode = INTERVENTIONS[intervention]["part_mode"]

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=config_path, version_base=None)
    cfg = hydra.compose(config_name=config_name, overrides=cfg_overrides or [])

    # Recover the trained feature layer from the checkpoint so it need not be
    # re-specified on the command line (and to catch a silent layer mismatch).
    ckpt = torch.load(ckpt_path, weights_only=False)
    sync_backbone_layer(cfg, ckpt, cfg_overrides or [])

    device = torch.device("cuda")
    test_loader = build_loader(cfg.dataset, "test", cfg.batch_size, 1)
    max_depth = test_loader.dataset.max_depth

    model = instantiate(cfg.backbone).to(device)
    probe = instantiate(cfg.probe, feat_dim=model.feat_dim, max_depth=max_depth).to(device)
    probe.load_state_dict(ckpt["probe"])
    model.eval()
    probe.eval()

    # ------------------------------------------------------------------
    # Build the subspace basis V_k (skipped for the full-rank baseline).
    # ------------------------------------------------------------------
    V_k = None
    if part_mode != "full":
        assert LINEAR_WEIGHT_KEY in ckpt["probe"], (
            f"Subspace intervention requires a linear probe head with weight key "
            f"'{LINEAR_WEIGHT_KEY}', not found in checkpoint. Got keys: "
            f"{list(ckpt['probe'].keys())[:4]}..."
        )
        W = ckpt["probe"][LINEAR_WEIGHT_KEY]  # [C, D, 1, 1]
        W_mat = W.view(W.shape[0], -1)        # [C, D]
        V_k = compute_subspace(W_mat, k, mode, shift).to(device)
        logger.info(
            f"intervention={intervention} (mode={mode}, part_mode={part_mode}) "
            f"| W=[{W_mat.shape[0]}, {W_mat.shape[1]}] | V_k=[{V_k.shape[0]}, {V_k.shape[1]}]"
        )
    else:
        logger.info("intervention=full (no subspace projection; full-rank upper bound)")

    # ------------------------------------------------------------------
    # Evaluation: scale-aware and scale-invariant depth metrics.
    # ------------------------------------------------------------------
    loss_fn = DepthLoss()

    sa_loss, sa_metrics = validate_depth(
        model, probe, test_loader, loss_fn, V_k, layer=layer, scale_invariant=False,
        save_pred=save_pred, part_mode=part_mode, energy_ctrl=energy_ctrl,
    )
    logger.info(f"Scale-Aware Loss: {sa_loss:.4f}")
    for mk, mv in sa_metrics.items():
        logger.info(f"SA {mk:10s}: {mv:.4f}")

    si_loss, si_metrics = validate_depth(
        model, probe, test_loader, loss_fn, V_k, layer=layer, scale_invariant=True,
        save_pred=False, part_mode=part_mode, energy_ctrl=energy_ctrl,
    )
    logger.info(f"Scale-Invariant Loss: {si_loss:.4f}")
    for mk, mv in si_metrics.items():
        logger.info(f"SI {mk:10s}: {mv:.4f}")

    return sa_metrics, si_metrics


def parse_args():
    p = argparse.ArgumentParser(
        description="Subspace intervention for a frozen depth probe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True, help="path to a trained probe checkpoint (.pth)")
    p.add_argument(
        "--intervention", default="aligned", choices=list(INTERVENTIONS),
        help="aligned (Eq.3) | residual (Eq.4) | random (R_k) | shifted | full",
    )
    p.add_argument("--k", type=int, default=32, help="subspace rank")
    p.add_argument("--shift", type=int, default=0, help="start index for --intervention shifted")
    p.add_argument(
        "--energy-ctrl", default="none", choices=["none", "match_full"],
        help="rescale intervened features to match the full-feature energy",
    )
    p.add_argument("--save-pred", action="store_true", help="save predicted depth visualizations")
    p.add_argument("--layer", type=int, default=None, help="layer id, used only for save-pred paths")
    p.add_argument("--config-path", default="./configs")
    p.add_argument("--config-name", default="depth_training")
    p.add_argument(
        "--overrides", default="",
        help="space-separated hydra overrides, e.g. "
             "\"backbone=dinov2_l14 probe=depth_linear ++backbone.layer=17\"",
    )
    return p.parse_args()


def main():
    args = parse_args()
    evaluate(
        ckpt_path=args.ckpt,
        intervention=args.intervention,
        k=args.k,
        shift=args.shift,
        energy_ctrl=args.energy_ctrl,
        save_pred=args.save_pred,
        layer=args.layer,
        config_path=args.config_path,
        config_name=args.config_name,
        cfg_overrides=shlex.split(args.overrides),
    )


if __name__ == "__main__":
    main()


"""
Example:
    python eval_subspace.py --intervention aligned --ckpt path/to/ckpt.pth --k 8 \
        --overrides "backbone=dinov2_l14 probe=depth_linear ++backbone.layer=17 \
        +backbone.return_multilayer=False system.num_gpus=1 batch_size=1 use_alignment=False"
"""
