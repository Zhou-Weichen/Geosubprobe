from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.functional import interpolate
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from evals.datasets.builder import build_loader
from evals.utils.optim import cosine_decay_linear_warmup


def compute_intersection_union(pred_logits, target, num_classes=41, ignore_index=0):
    pred_labels = pred_logits.argmax(dim=1)  
    valid_mask = (target != ignore_index)   
    
    pred_labels = pred_labels[valid_mask]
    target = target[valid_mask]
    
    intersect = pred_labels[pred_labels == target]
    
    area_intersect = torch.histc(intersect.float(), bins=num_classes, min=0, max=num_classes-1)
    area_pred = torch.histc(pred_labels.float(), bins=num_classes, min=0, max=num_classes-1)
    area_target = torch.histc(target.float(), bins=num_classes, min=0, max=num_classes-1)
    area_union = area_pred + area_target - area_intersect
    
    correct_pixels = intersect.numel()
    valid_pixels = valid_mask.sum().item()
    
    return area_intersect, area_union, correct_pixels, valid_pixels


def ddp_setup(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def train(
    model,
    probe,
    train_loader,
    optimizer,
    scheduler,
    n_epochs,
    detach_model,
    loss_fn,
    rank=0,
    world_size=1,
    valid_loader=None,
):
    for ep in range(n_epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(ep)

        train_loss = 0
        pbar = tqdm(train_loader) if rank == 0 else train_loader
        for i, batch in enumerate(pbar):
            images = batch["image"].to(rank)
            target = batch["mask"].to(rank)  

            optimizer.zero_grad()
            if detach_model:
                with torch.no_grad():
                    feats = model(images)
                    if isinstance(feats, (tuple, list)):
                        feats = [_f.detach() for _f in feats]
                    else:
                        feats = feats.detach()
            else:
                feats = model(images)
                
            pred = probe(feats) 
            pred = interpolate(
                pred, size=target.shape[-2:], mode="bilinear", align_corners=True
            )

            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            pr_lr = optimizer.param_groups[0]["lr"]
            loss = loss.item()
            train_loss += loss

            if rank == 0:
                _loss = train_loss / (i + 1)
                pbar.set_description(
                    f"{ep} | loss: {loss:.4f} ({_loss:.4f}) probe_lr: {pr_lr:.2e}"
                )

        train_loss /= len(train_loader)

        if rank == 0:
            logger.info(f"train loss {ep}   | {train_loss:.4f}")
            if valid_loader is not None:
                val_loss, val_metrics = validate(
                    model, probe, valid_loader, loss_fn, num_classes=41, ignore_index=0
                )
                logger.info(f"valid loss {ep}   | {val_loss:.4f}")
                logger.info(f"valid mIoU {ep}   | {val_metrics['mIoU']:.4f}")
                logger.info(f"valid mAcc {ep}   | {val_metrics['mAcc']:.4f}")
                logger.info(f"valid aAcc {ep}   | {val_metrics['aAcc']:.4f}")


def validate(
    model, probe, loader, loss_fn, verbose=True, num_classes=41, ignore_index=0
):
    total_loss = 0.0
    
    total_area_intersect = torch.zeros(num_classes, dtype=torch.float64).cuda()
    total_area_union = torch.zeros(num_classes, dtype=torch.float64).cuda()
    total_area_target = torch.zeros(num_classes, dtype=torch.float64).cuda()
    total_correct_pixels = 0
    total_valid_pixels = 0

    with torch.inference_mode():
        pbar = tqdm(loader, desc="Evaluation") if verbose else loader
        for batch in pbar:
            images = batch["image"].cuda()
            target = batch["mask"].cuda()

            feat = model(images)
            pred = probe(feat).detach()
            pred = interpolate(
                pred, size=target.shape[-2:], mode="bilinear", align_corners=True
            )

            loss = loss_fn(pred, target)
            total_loss += loss.item()

            area_intersect, area_union, correct_px, valid_px = compute_intersection_union(
                pred, target, num_classes=num_classes, ignore_index=ignore_index
            )
            
            total_area_intersect += area_intersect.cuda()
            total_area_union += area_union.cuda()
            area_target = torch.histc(target.float()[target != ignore_index], bins=num_classes, min=0, max=num_classes-1)
            total_area_target += area_target.cuda()
            
            total_correct_pixels += correct_px
            total_valid_pixels += valid_px


    total_loss = total_loss / len(loader)
    
    iou_per_class = total_area_intersect[1:] / (total_area_union[1:] + 1e-10)
    acc_per_class = total_area_intersect[1:] / (total_area_target[1:] + 1e-10)
    
    metrics = {
        "mIoU": (iou_per_class.mean() * 100).item(),                  # Mean IoU
        "mAcc": (acc_per_class.mean() * 100).item(),                  # Mean Class Accuracy
        "aAcc": (total_correct_pixels / (total_valid_pixels + 1e-10)) * 100  # All Pixel Accuracy
    }

    return total_loss, metrics


def train_model(rank, world_size, cfg):
    if world_size > 1:
        ddp_setup(rank, world_size, cfg.system.port)

    trainval_loader = build_loader(cfg.dataset, "train", cfg.batch_size, world_size) 
    test_loader = build_loader(cfg.dataset, "test", cfg.batch_size, 1)

    model = instantiate(cfg.backbone)

    probe = instantiate(
        cfg.probe, feat_dim=model.feat_dim, output_dim=41
    )

    timestamp = datetime.now().strftime("%d%m%Y-%H%M")
    train_dset = trainval_loader.dataset.name
    test_dset = test_loader.dataset.name
    model_info = [
        f"{model.checkpoint_name:40s}",
        f"{model.patch_size:2d}",
        f"{str(model.layer):5s}",
        f"{model.output:10s}",
    ]
    probe_info = [f"{probe.name:25s}"]
    batch_size = cfg.batch_size * cfg.system.num_gpus
    train_info = [
        f"{cfg.optimizer.n_epochs:3d}",
        f"{cfg.optimizer.warmup_epochs:4.2f}",
        f"{str(cfg.optimizer.probe_lr):>10s}",
        f"{str(cfg.optimizer.model_lr):>10s}",
        f"{batch_size:4d}",
        f"{train_dset:10s}",
        f"{test_dset:10s}",
    ]

    exp_name = "_".join([timestamp] + model_info + probe_info + train_info)
    exp_name = f"{exp_name}_{cfg.note}" if cfg.note != "" else exp_name
    exp_name = exp_name.replace(" ", "")

    if rank == 0:
        exp_path = Path(__file__).parent / f"semantic_exps/{exp_name}"
        exp_path.mkdir(parents=True, exist_ok=True)
        logger.add(exp_path / "training.log")
        logger.info(f"Config: \n {OmegaConf.to_yaml(cfg)}")

    model = model.to(rank)
    probe = probe.to(rank)

    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        probe = DDP(probe, device_ids=[rank])

    if cfg.optimizer.model_lr == 0:
        optimizer = torch.optim.AdamW(
            [{"params": probe.parameters(), "lr": cfg.optimizer.probe_lr}]
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": probe.parameters(), "lr": cfg.optimizer.probe_lr},
                {"params": model.parameters(), "lr": cfg.optimizer.model_lr},
            ]
        )

    lambda_fn = lambda epoch: cosine_decay_linear_warmup(  # noqa: E731
        epoch,
        cfg.optimizer.n_epochs * len(trainval_loader),
        cfg.optimizer.warmup_epochs * len(trainval_loader),
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_fn)
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0).to(rank)

    train(
        model,
        probe,
        trainval_loader,
        optimizer,
        scheduler,
        cfg.optimizer.n_epochs,
        detach_model=(cfg.optimizer.model_lr == 0),
        loss_fn=loss_fn,
        rank=rank,
        world_size=world_size,
    )

    if rank == 0:
        logger.info(f"Evaluating on test split of {test_dset}")

        test_loss, test_metrics = validate(model, probe, test_loader, loss_fn, num_classes=41, ignore_index=0)
        logger.info(f"Final test loss       | {test_loss:.4f}")
        for metric in test_metrics:
            logger.info(f"Final test {metric:10s} | {test_metrics[metric]:.4f}")
            
        results = ", ".join([f"{test_metrics[_m]:.4f}" for _m in test_metrics])

        exp_info = ", ".join(model_info + probe_info + train_info)
        log = f"{timestamp}, {exp_info}, {results} \n"
        with open(f"semantic_results_{test_dset}.log", "a") as f:
            f.write(log)

        ckpt_path = exp_path / "ckpt.pth"
        checkpoint = {
            "cfg": cfg,
            "model": model.module.state_dict() if world_size > 1 else model.state_dict(),
            "probe": probe.module.state_dict() if world_size > 1 else probe.state_dict(),
        }
        torch.save(checkpoint, ckpt_path)
        logger.info(f"Saved checkpoint at {ckpt_path}")

    if world_size > 1:
        destroy_process_group()


@hydra.main(config_name="semantic_training", config_path="./configs", version_base=None)
def main(cfg: DictConfig):
    world_size = cfg.system.num_gpus
    if world_size > 1:
        mp.spawn(train_model, args=(world_size, cfg), nprocs=world_size)
    else:
        train_model(0, world_size, cfg)


if __name__ == "__main__":
    main()