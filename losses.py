import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Multi-class Dice loss (soft dice) for segmentation.
    logits: (B, C, H, W)
    target: (B, H, W) long in [0..C-1]
    """
    def __init__(self, num_classes: int, smooth: float = 1.0, ignore_index: int | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        assert C == self.num_classes

        # mask ignore pixels if needed
        if self.ignore_index is not None:
            valid = (target != self.ignore_index)
            target = target.clone()
            target[~valid] = 0  # placeholder
        else:
            valid = None

        probs = F.softmax(logits, dim=1)  # (B,C,H,W)
        target_1h = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()  # (B,C,H,W)

        if valid is not None:
            valid = valid.unsqueeze(1).float()  # (B,1,H,W)
            probs = probs * valid
            target_1h = target_1h * valid

        dims = (0, 2, 3)  # sum over batch and spatial
        intersection = torch.sum(probs * target_1h, dims)
        denom = torch.sum(probs + target_1h, dims)

        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)  # (C,)
        loss = 1.0 - dice
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Multi-class focal loss (Lin et al., 2017).
    Focal loss down-weights easy (confident) predictions and focuses training
    on hard/rare examples — more effective than extreme class weights for
    severely imbalanced segmentation tasks.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=2 is the standard default; higher values increase focus on hard pixels.
    class_weights acts as alpha_t (per-class prior balancing).
    """

    def __init__(self, num_classes: int, gamma: float = 2.0,
                 class_weights: torch.Tensor | None = None,
                 label_smoothing: float = 0.0,
                 ignore_index: int | None = None):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index if ignore_index is not None else -100
        self.register_buffer("class_weights", class_weights)

    def _pixel_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-pixel focal loss (B,H,W) without reduction. Used by OHEM."""
        log_p = F.log_softmax(logits, dim=1)
        p     = log_p.exp()

        t      = target.unsqueeze(1).clamp(min=0)
        log_pt = log_p.gather(1, t).squeeze(1)
        pt     = p.gather(1, t).squeeze(1)

        if self.label_smoothing > 0:
            smooth_loss = -log_p.mean(dim=1)
            log_pt = (1 - self.label_smoothing) * log_pt + self.label_smoothing * (-smooth_loss)

        loss = -(1.0 - pt) ** self.gamma * log_pt     # (B,H,W)

        if self.class_weights is not None:
            loss = self.class_weights[target.clamp(min=0)] * loss

        return loss

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self._pixel_loss(logits, target)
        if self.ignore_index >= 0:
            loss = loss[target != self.ignore_index]
        return loss.mean()


class FocalDiceLoss(nn.Module):
    """
    Focal loss + Dice loss, with optional OHEM on the focal component.

    ohem_ratio: fraction of hardest pixels to keep (1.0 = disabled).
    OHEM selects the top-K loss pixels by focal value, forcing gradient signal
    onto rare/hard classes instead of averaging with easy background pixels.
    """

    def __init__(self, num_classes: int, dice_weight: float = 0.85,
                 gamma: float = 2.0,
                 class_weights: torch.Tensor | None = None,
                 label_smoothing: float = 0.0,
                 ignore_index: int | None = None,
                 ohem_ratio: float = 1.0):
        super().__init__()
        self.focal = FocalLoss(num_classes=num_classes, gamma=gamma,
                               class_weights=class_weights,
                               label_smoothing=label_smoothing,
                               ignore_index=ignore_index)
        self.dice        = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.ohem_ratio  = ohem_ratio
        self._ignore     = self.focal.ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.ohem_ratio < 1.0:
            pixel_loss = self.focal._pixel_loss(logits, target)   # (B,H,W)
            flat = pixel_loss.flatten()
            if self._ignore >= 0:
                flat = flat[(target != self._ignore).flatten()]
            k = max(1, int(flat.numel() * self.ohem_ratio))
            focal_loss = flat.topk(k).values.mean()
        else:
            focal_loss = self.focal(logits, target)

        return focal_loss + self.dice_weight * self.dice(logits, target)


class CEDiceLoss(nn.Module):
    def __init__(self, num_classes: int, dice_weight: float = 0.7, ce_weight: float = 1.0,
                 ignore_index: int | None = None, class_weights: torch.Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index if ignore_index is not None else -100,
            label_smoothing=label_smoothing,
        )
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, target)
        loss_dice = self.dice(logits, target)
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice
