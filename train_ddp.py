#!/usr/bin/env python
"""
Face-parsing DDP training.

Launch:
    torchrun --nproc_per_node=4 train_ddp.py [options]

Examples:
    # LiteUNetV2 base=64 (default)
    torchrun --nproc_per_node=4 train_ddp.py

    # UNetV2 base=32, lower LR
    torchrun --nproc_per_node=4 train_ddp.py --model unetv2 --base 32 --lr 1e-2

    # Quick experiment: fewer epochs, no OHEM
    torchrun --nproc_per_node=4 train_ddp.py --epochs 100 --ohem-ratio 1.0
"""
import os, sys, time, shutil, logging, argparse
from pathlib import Path
from contextlib import contextmanager

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import wandb

from unet import UNetV2, LiteUNetV2
from losses import FocalDiceLoss
from metrics import confusion_matrix, f1_macro_from_cm, fbeta_present_gt_from_cm
from palette import NUM_CLASSES, rgb_to_label
from dataset import FaceParsingDataset
from split_utils import list_images, make_split, save_split
from augment import make_face_aug

logging.getLogger("torch._inductor").setLevel(logging.WARNING)
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)

# ── Defaults ──────────────────────────────────────────────────────────────────

_SRC_IMG_DIR  = "train/images"
_SRC_MASK_DIR = "train/masks"
_TMP_ROOT     = "/tmp/facemask"
_HARD_CLASSES = {14, 15, 16}
_FLIP_PAIRS   = [(4, 5), (6, 7), (8, 9)]


def _parse_args():
    p = argparse.ArgumentParser(description="Face-parsing DDP training")

    # ── Data / infra ───────────────────────────────────────────────────────────
    p.add_argument("--img-dir",       default=_SRC_IMG_DIR)
    p.add_argument("--mask-dir",      default=_SRC_MASK_DIR)
    p.add_argument("--tmp-root",      default=_TMP_ROOT)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--val-ratio",     type=float, default=0.1)

    # ── Model ──────────────────────────────────────────────────────────────────
    p.add_argument("--model",    choices=["unetv2", "liteunetv2"], default="liteunetv2",
                   help="unetv2: standard DoubleConv encoder; liteunetv2: DSConv encoder (wider for same budget)")
    p.add_argument("--base",     type=int,   default=64,
                   help="Base channel width. LiteUNetV2 default=64 (~1.8M params); UNetV2 default=23 (~1.8M params)")
    p.add_argument("--dropout",  type=float, default=0.3)

    # ── Training ───────────────────────────────────────────────────────────────
    p.add_argument("--batch-per-gpu",  type=int,   default=16)
    p.add_argument("--epochs",         type=int,   default=300)
    p.add_argument("--lr",             type=float, default=1.5e-2)
    p.add_argument("--warmup-epochs",  type=int,   default=20,
                   help="Linear warmup length in epochs (calibrated to ~280 gradient steps with bs=64)")
    p.add_argument("--weight-decay",   type=float, default=1e-4)
    p.add_argument("--eta-min",        type=float, default=5e-5)

    # ── Loss ───────────────────────────────────────────────────────────────────
    p.add_argument("--focal-gamma",      type=float, default=2.0)
    p.add_argument("--dice-w",           type=float, default=0.85)
    p.add_argument("--aux-weight",       type=float, default=0.4)
    p.add_argument("--label-smoothing",  type=float, default=0.05)
    p.add_argument("--ohem-ratio",       type=float, default=0.7,
                   help="Fraction of hardest pixels kept by OHEM (1.0 = disabled)")

    # ── Sampler / hard classes ─────────────────────────────────────────────────
    p.add_argument("--hard-class-weight", type=float, default=3.0,
                   help="Sampling multiplier for images containing hat/ear_r/neck_l")

    # ── Misc ───────────────────────────────────────────────────────────────────
    p.add_argument("--ema-decay",  type=float, default=0.999)
    p.add_argument("--ckpt-path",  default="checkpoints/best_ddp.pt")

    return p.parse_args()

# ── /tmp copy ─────────────────────────────────────────────────────────────────

def _copy_to_tmp(src: str, dst: str) -> str:
    if Path(dst).exists():
        return dst
    shutil.copytree(src, dst)
    return dst

# ── EMA ───────────────────────────────────────────────────────────────────────

class EMAKeeper:
    """
    Operates on the raw nn.Module (not DDP/compiled wrapper) so parameter
    names have no 'module.' prefix and state dicts are checkpoint-compatible.
    """

    def __init__(self, module: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.num_updates = 0
        self.shadow = {n: p.data.clone() for n, p in module.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, module: nn.Module):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for n, p in module.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(decay).add_(p.data, alpha=1 - decay)

    @contextmanager
    def applied(self, module: nn.Module):
        """Swap raw module weights with EMA shadow in-place for evaluation."""
        original = {n: p.data.clone() for n, p in module.named_parameters() if p.requires_grad}
        for n, p in module.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])
        try:
            yield
        finally:
            for n, p in module.named_parameters():
                if p.requires_grad:
                    p.data.copy_(original[n])

# ── Validate ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, raw_model, loader, device, amp_dtype, ema):
    """
    Each rank evaluates its shard of val data.
    Confusion matrices are all_reduced (summed) across ranks before computing F1.
    EMA applied to raw_model — DDP reads through to the same weights.
    """
    ctx = ema.applied(raw_model) if ema is not None else contextmanager(lambda: (yield))()
    model.eval()
    cm_total = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device)

    with ctx:
        for imgs, masks, _ in loader:
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits      = model(imgs)
                logits_flip = model(imgs.flip(-1)).flip(-1)
                for a, b in _FLIP_PAIRS:
                    logits_flip[:, [a, b]] = logits_flip[:, [b, a]]
                logits = (logits + logits_flip) * 0.5

            pred      = logits.argmax(dim=1)
            cm_total += confusion_matrix(pred, masks, num_classes=NUM_CLASSES).to(device)

    dist.all_reduce(cm_total, op=dist.ReduceOp.SUM)
    cm_cpu = cm_total.cpu()
    macro_f1, per_class_f1 = f1_macro_from_cm(cm_cpu)
    val_fscore, _, _ = fbeta_present_gt_from_cm(cm_cpu, beta=1.0)
    return macro_f1, per_class_f1, val_fscore

# ── Downsample mask ───────────────────────────────────────────────────────────

def _downsample_mask(masks: torch.Tensor, size: tuple) -> torch.Tensor:
    return F.interpolate(masks.float().unsqueeze(1), size=size, mode="nearest").squeeze(1).long()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    is_main    = (rank == 0)
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True

    # ── /tmp: rank 0 copies, others wait ──────────────────────────────────────
    if is_main:
        img_dir  = _copy_to_tmp(args.img_dir,  f"{args.tmp_root}/images")
        mask_dir = _copy_to_tmp(args.mask_dir, f"{args.tmp_root}/masks")
        print(f"Data: {img_dir}  {mask_dir}")
    dist.barrier()
    img_dir  = f"{args.tmp_root}/images"
    mask_dir = f"{args.tmp_root}/masks"

    # ── Split ─────────────────────────────────────────────────────────────────
    all_files = list_images(img_dir)
    train_files, val_files = make_split(all_files, val_ratio=args.val_ratio, seed=args.seed)
    if is_main:
        save_split(train_files, val_files, out_dir="splits", tag=f"seed{args.seed}_vr{args.val_ratio}")
        print(f"Split: {len(train_files)} train / {len(val_files)} val")

    # ── Class weights (all ranks compute identically — same inputs, same result) ──
    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    mask_lookup  = {p.stem: p for p in Path(mask_dir).iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    for fn in train_files:
        stem     = Path(fn).stem
        mask_rgb = np.array(PILImage.open(mask_lookup[stem]).convert("RGB"), dtype=np.uint8)
        for c, n in enumerate(np.bincount(rgb_to_label(mask_rgb).ravel(), minlength=NUM_CLASSES)):
            pixel_counts[c] += n
    freq             = pixel_counts / pixel_counts.sum()
    median_freq      = float(np.median(freq[freq > 0]))
    class_weights_np = np.where(freq > 0, np.sqrt(median_freq / freq), 1.0)
    class_weights    = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

    # ── Datasets ──────────────────────────────────────────────────────────────
    aug_fn   = make_face_aug(p_flip=0.5, p_geom=0.7, p_color=0.7, p_blur=0.15)
    train_ds = FaceParsingDataset(img_dir, mask_dir, file_list=train_files, augment=aug_fn, cache=True)
    val_ds   = FaceParsingDataset(img_dir, mask_dir, file_list=val_files,   augment=None,   cache=True)

    # ── Samplers ──────────────────────────────────────────────────────────────
    sample_weights = torch.tensor([
        args.hard_class_weight if any(np.any(lbl == c) for c in _HARD_CLASSES) else 1.0
        for lbl in train_ds._mask_cache
    ], dtype=torch.float)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed + rank)
    train_sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_ds) // world_size,
        replacement=True,
        generator=train_generator,
    )
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank,
                                     shuffle=False, drop_last=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_per_gpu, sampler=train_sampler,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds,   batch_size=16, sampler=val_sampler,
                              num_workers=2, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cls = LiteUNetV2 if args.model == "liteunetv2" else UNetV2
    raw_model = model_cls(num_classes=NUM_CLASSES, base=args.base,
                          dropout=args.dropout, deep_supervision=True).to(device)
    ddp_model = DDP(raw_model, device_ids=[local_rank])
    try:
        model = torch.compile(ddp_model, mode="default")
        if is_main: print("torch.compile: enabled")
    except Exception as e:
        model = ddp_model
        if is_main: print(f"torch.compile: skipped ({e})")

    num_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    ema        = EMAKeeper(raw_model, decay=args.ema_decay)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer       = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                        weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler       = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.eta_min),
        ],
        milestones=[args.warmup_epochs],
    )

    use_bf16  = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler    = torch.amp.GradScaler("cuda", enabled=not use_bf16)

    criterion = FocalDiceLoss(
        num_classes=NUM_CLASSES, dice_weight=args.dice_w, gamma=args.focal_gamma,
        class_weights=class_weights, label_smoothing=args.label_smoothing,
        ohem_ratio=args.ohem_ratio,
    )

    # ── wandb (rank 0 only) ───────────────────────────────────────────────────
    effective_batch = args.batch_per_gpu * world_size
    model_tag = f"{args.model}_b{args.base}"
    if is_main:
        wandb.init(
            project="face-parsing-unet",
            name=f"{model_tag}_ddp{world_size}_bs{effective_batch}_lr{args.lr}",
            config={
                "model": args.model, "num_classes": NUM_CLASSES,
                "base_width": args.base, "dropout": args.dropout, "params": num_params,
                "loss": "FocalDice+OHEM", "focal_gamma": args.focal_gamma,
                "dice_weight": args.dice_w, "ohem_ratio": args.ohem_ratio,
                "aux_weight": args.aux_weight,
                "class_weights": "sqrt(median/freq)",
                "hard_classes": sorted(_HARD_CLASSES),
                "hard_class_weight": args.hard_class_weight,
                "lr": args.lr, "warmup_epochs": args.warmup_epochs,
                "epochs": args.epochs, "batch_per_gpu": args.batch_per_gpu,
                "effective_batch": effective_batch, "world_size": world_size,
                "ema_decay": args.ema_decay, "steps_per_epoch": steps_per_epoch,
                "amp_dtype": str(amp_dtype),
                "augmentation": "hflip+label_swap, ShiftScaleRotate, ColorJitter, GaussianBlur",
            },
        )
        print(f"DDP      : {world_size} GPUs  |  batch/GPU={args.batch_per_gpu}  effective_batch={effective_batch}")
        print(f"Model    : {model_cls.__name__}(base={args.base})  |  Params: {num_params:,}")
        print(f"AMP      : {amp_dtype}")
        print(f"Training : {args.epochs} epochs × {steps_per_epoch} steps/epoch")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_f1 = -1.0
    os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0      = time.time()
        running = 0.0

        for imgs, masks, _ in train_loader:
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs = model(imgs)
                if isinstance(outputs, tuple):
                    main_out, aux1, aux2 = outputs
                    loss = (criterion(main_out, masks)
                            + args.aux_weight * criterion(aux1, _downsample_mask(masks, aux1.shape[-2:]))
                            + args.aux_weight * criterion(aux2, _downsample_mask(masks, aux2.shape[-2:])))
                else:
                    loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(raw_model)
            running += loss.item()

        scheduler.step()
        val_f1, per_class_f1, val_fscore = validate(model, raw_model, val_loader, device, amp_dtype, ema)
        elapsed    = time.time() - t0
        train_loss = running / max(1, len(train_loader))
        cur_lr     = scheduler.get_last_lr()[0]

        if is_main:
            wandb.log({"epoch": epoch, "train/loss": train_loss, "val/macro_f1": val_f1,
                       "val/fscore": val_fscore, "lr": cur_lr, "time/epoch_sec": elapsed,
                       "ema/decay": min(args.ema_decay, (1 + ema.num_updates) / (10 + ema.num_updates))})
            print(f"[{epoch:03d}/{args.epochs}] loss={train_loss:.4f}  val_F1={val_f1:.4f}"
                  f"  fscore={val_fscore:.4f}  lr={cur_lr:.2e}  time={elapsed:.1f}s")

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save({
                    "model":        raw_model.state_dict(),
                    "ema_shadow":   ema.shadow,
                    "optimizer":    optimizer.state_dict(),
                    "epoch":        epoch,
                    "best_f1":      best_f1,
                    "per_class_f1": per_class_f1.tolist(),
                    "args":         vars(args),
                }, args.ckpt_path)
                print(f"  saved {args.ckpt_path}  (F1={best_f1:.4f})")

    if is_main:
        print(f"\nBest val macro-F1: {best_f1:.4f}")
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
