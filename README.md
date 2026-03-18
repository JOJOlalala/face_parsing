# Face Parsing Segmentation

Semantic segmentation of facial images into **19 classes** (skin, eyes, hair, etc.) using U-Net variants trained on ~1,000 face images.

---

## Table of Contents

- [Classes](#classes)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Model Architecture](#model-architecture)
- [Checkpoints](#checkpoints)

---

## Classes

| ID | Name | Color (RGB) |
|----|------|-------------|
| 0  | background | (0, 0, 0) |
| 1  | skin | (204, 0, 0) |
| 2  | nose | (76, 153, 0) |
| 3  | eye_g (eyeglasses) | (204, 204, 0) |
| 4  | l_eye | (51, 51, 255) |
| 5  | r_eye | (204, 0, 204) |
| 6  | l_brow | (0, 255, 255) |
| 7  | r_brow | (255, 204, 204) |
| 8  | l_ear | (102, 51, 0) |
| 9  | r_ear | (255, 0, 0) |
| 10 | mouth | (102, 204, 0) |
| 11 | u_lip | (255, 255, 0) |
| 12 | l_lip | (0, 0, 153) |
| 13 | hair | (0, 0, 204) |
| 14 | hat | (255, 51, 153) |
| 15 | ear_r (earring) | (0, 204, 204) |
| 16 | neck_l (necklace) | (0, 51, 0) |
| 17 | neck | (255, 153, 51) |
| 18 | cloth | (0, 204, 0) |

Classes 14, 15, 16 (hat, earring, necklace) are **hard classes** — rare and small, requiring special handling during training.

---

## Project Structure

```
facemask/
├── train/
│   ├── images/          # 1000 RGB face images
│   └── masks/           # Corresponding RGB color-encoded label maps
├── test/
│   ├── images/          # Test images for inference
│   └── masks/           # Predicted segmentation masks (output)
├── checkpoints/         # Saved model weights
├── splits/              # Train/val split files
│   ├── train_seed42_vr0.1.txt
│   └── val_seed42_vr0.1.txt
├── wandb/               # Experiment tracking logs
├── train_ddp.py         # Main multi-GPU training script
├── train_split.ipynb    # Single-GPU training notebook
├── unet.py              # Model architectures
├── dataset.py           # Dataset and data loading
├── losses.py            # Loss functions (Focal, Dice, OHEM)
├── augment.py           # Data augmentation pipeline
├── metrics.py           # Evaluation metrics (macro-F1)
├── palette.py           # Class definitions and color mapping
├── split_utils.py       # Train/val split utilities
├── inference.ipynb      # Inference and visualization notebook
└── ablation.ipynb       # Ablation study results
```

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

### Multi-GPU (Recommended)

Launch with `torchrun` for DDP (DistributedDataParallel) training across all available GPUs:

```bash
torchrun --nproc_per_node=4 train_ddp.py \
    --model unetv2 \
    --base 23 \
    --epochs 300 \
    --batch_size 16 \
    --lr 1.5e-2 \
    --warmup_epochs 20 \
    --dropout 0.3 \
    --ohem_ratio 0.7 \
    --ema_decay 0.999 \
    --deep_supervision \
    --cache
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `unetv2` | Architecture: `unetv2`, `liteunetv2`, `unet` |
| `--base` | `23` | Base channel width |
| `--epochs` | `300` | Total training epochs |
| `--batch_size` | `16` | Per-GPU batch size (effective = batch × GPUs) |
| `--lr` | `1.5e-2` | Peak learning rate |
| `--warmup_epochs` | `20` | Linear LR warmup epochs |
| `--dropout` | `0.3` | Dropout2d rate after bottleneck |
| `--ohem_ratio` | `0.7` | OHEM: fraction of hardest pixels used for loss |
| `--ema_decay` | `0.999` | EMA decay for validation weights |
| `--deep_supervision` | off | Enable auxiliary losses at H/8 and H/4 |
| `--cache` | off | Cache dataset in RAM (~750MB) for faster I/O |

### Single-GPU

Open and run [`train_split.ipynb`](train_split.ipynb) for an interactive single-GPU training loop.

### Training Details

**Data split:** 900 train / 100 val (random seed 42, 10% val ratio)

**Augmentation pipeline:**
1. Horizontal flip with symmetric label swapping (`l_eye↔r_eye`, `l_brow↔r_brow`, `l_ear↔r_ear`) — p=0.5
2. ShiftScaleRotate (shift±3%, scale±20%, rotate±10°) — p=0.7
3. ColorJitter (brightness 0.3, contrast 0.2, saturation 0.15, hue 0.05) — p=0.7
4. GaussianBlur (kernel 3–5) — p=0.15

**Class balancing:**
- Loss weights: sqrt median-frequency balancing over all 19 classes
- Sampler: hard-class images (hat/earring/necklace) upsampled 3×

**Loss function:** `FocalDiceLoss` + OHEM
```
loss = FocalLoss(γ=2.0, label_smoothing=0.05)
     + 0.85 × DiceLoss
     [applied to top-70% hardest pixels per OHEM]

total = main_loss + 0.4 × aux1_loss@H/8 + 0.4 × aux2_loss@H/4
```

**Optimizer & scheduler:**
- AdamW (lr=1.5e-2, weight_decay=1e-4)
- Linear warmup → CosineAnnealingLR (eta_min=5e-5)
- Gradient clipping: max_norm=1.0

**Mixed precision:** bfloat16 (or float16 with GradScaler)

Checkpoints are saved to `checkpoints/` at each epoch where validation macro-F1 improves.

---

## Inference

Open [`inference.ipynb`](inference.ipynb) for step-by-step inference and visualization.

### Quick Start (Python)

```python
import torch
from unet import UNetV2
from dataset import FaceParsingDataset
from torch.utils.data import DataLoader

device = torch.device("cuda")

# --- Load model ---
ckpt = torch.load("checkpoints/unetv2_full.pt", weights_only=False)
model = UNetV2(num_classes=19, base=23, dropout=0.3).to(device)

# Strip torch.compile() prefix if present
state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in ckpt["model"].items()}
model.load_state_dict(state_dict)

# Use EMA weights for better generalization
if "ema_shadow" in ckpt:
    for name, param in model.named_parameters():
        if name in ckpt["ema_shadow"]:
            param.data.copy_(ckpt["ema_shadow"][name])

model.eval()

# --- Predict with optional TTA ---
FLIP_PAIRS = [(4, 5), (6, 7), (8, 9)]  # (l_eye,r_eye), (l_brow,r_brow), (l_ear,r_ear)

@torch.no_grad()
def predict(imgs, use_tta=True):
    imgs = imgs.to(device)
    with torch.amp.autocast("cuda"):
        logits = model(imgs)

    if use_tta:
        logits_flip = model(torch.flip(imgs, dims=[-1]))
        logits_flip = torch.flip(logits_flip, dims=[-1])
        for a, b in FLIP_PAIRS:
            logits_flip[:, a], logits_flip[:, b] = logits_flip[:, b].clone(), logits_flip[:, a].clone()
        probs = (torch.softmax(logits, dim=1) + torch.softmax(logits_flip, dim=1)) * 0.5
        return probs.argmax(dim=1)

    return logits.argmax(dim=1)  # (B, H, W) — class indices 0–18

# --- Run inference ---
dataset = FaceParsingDataset("test/images")
loader = DataLoader(dataset, batch_size=4, shuffle=False)

for imgs, fnames in loader:
    preds = predict(imgs).cpu().numpy()  # (B, H, W)
    # preds[i] contains per-pixel class indices; save or visualize as needed
```

**Output:** Each prediction is a `(H, W)` array of integer class indices (0–18). Save as palette-mode PNG using the color mapping in `palette.py` for visualization.

### Test-Time Augmentation (TTA)

TTA averages softmax probabilities from the original image and its horizontal flip, with paired label channels swapped on the flipped result. This consistently improves macro-F1 (~+0.12%).

---

## Evaluation

Primary metric: **macro-F1** (unweighted average F1 over all 19 classes).

```python
from metrics import confusion_matrix, f1_macro_from_cm

cm = confusion_matrix(preds, targets, num_classes=19)
macro_f1 = f1_macro_from_cm(cm)
```

### Ablation Results (50 epochs, val split)

| Configuration | Macro-F1 |
|---------------|----------|
| Baseline (standard UNet) | 0.7656 |
| + UNetV2 architecture | 0.7774 |
| + Focal loss | 0.7793 |
| + Class weights | 0.7866 |
| + OHEM | 0.7841 |
| + Data augmentation | 0.7832 |
| + Deep supervision | 0.7870 |
| + EMA | 0.7871 |
| + TTA (hflip + label swap) | **0.7883** |

---

## Model Architecture

### UNetV2 (default)

```
Input (3, H, W)
  │
  ├─ Encoder: 4× DoubleConv blocks (stride-2 MaxPool)
  │             x1(H) → x2(H/2) → x3(H/4) → x4(H/8)
  │
  ├─ Bottleneck: LightASPP [1×1 | dilated r=6 | dilated r=12]
  │              + Dropout2d(0.3)     → xb(H/16)
  │
  └─ Decoder: 4× Up blocks with Attention Gates on skip connections
              d3(H/8) → d2(H/4) → d1(H/2) → d0(H)
                │           │
              aux_head1   aux_head2   (deep supervision, training only)
                          │
                      main_head → logits (19, H, W)
```

- **AttentionGate** — additive attention suppresses background activations on skip connections
- **LightASPP** — multi-scale context with depthwise-separable dilated convolutions (GroupNorm)
- **Deep supervision** — auxiliary 1×1 heads at H/8 and H/4 strengthen encoder gradients
- **EMA** — smoothed shadow weights used at validation/inference time

### LiteUNetV2

Same topology as UNetV2 but with depthwise-separable convolutions (DSConv) throughout encoder and decoder, reducing parameters ~4× for faster inference.

### Parameters

| Model | Base Width | Params |
|-------|-----------|--------|
| UNetV2 | 23 | ~1.8M |
| LiteUNetV2 | 48 | ~1.0M |
| UNet (baseline) | 32 | ~1.73M |

---

## Checkpoints

| File | Description |
|------|-------------|
| `checkpoints/unetv2_full.pt` | UNetV2 trained on full 1000 images |
| `checkpoints/v2_best_ddp.pt` | UNetV2, best validation macro-F1 (DDP run) |
| `checkpoints/liteunetv2_b64.pt` | LiteUNetV2, batch-64 run |
| `checkpoints/unet_best.pt` | Standard UNet baseline |

Each checkpoint contains:
- `model` — model state dict
- `ema_shadow` — EMA smoothed weights (use for inference)
- `optimizer` — optimizer state (for training resume)
- `epoch` — epoch at save time
- `best_f1` — best macro-F1 achieved
- `args` / `config` — hyperparameters used
