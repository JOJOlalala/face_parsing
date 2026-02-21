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


class CEDiceLoss(nn.Module):
    def __init__(self, num_classes: int, dice_weight: float = 0.7, ce_weight: float = 1.0,
                 ignore_index: int | None = None, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index if ignore_index is not None else -100)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, target)
        loss_dice = self.dice(logits, target)
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice
