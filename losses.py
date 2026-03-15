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

        probs      = F.softmax(logits, dim=1)      # (B,C,H,W)
        probs_flat = probs.reshape(B, C, -1)        # (B,C,N) — view
        tgt_flat   = target.reshape(B, -1).clamp(min=0)  # (B,N)

        if self.ignore_index is not None:
            valid    = (target != self.ignore_index).reshape(B, -1)  # (B,N)
            validf   = valid.float()
        else:
            valid = validf = None

        # intersection[b,c] = sum of probs[b,c,n] where target[b,n]==c
        # Avoids allocating (B,C,H,W) one_hot; uses gather + scatter_add instead.
        gt_prob = probs_flat.gather(1, tgt_flat.unsqueeze(1)).squeeze(1)  # (B,N)
        ones    = torch.ones_like(gt_prob)
        if validf is not None:
            gt_prob = gt_prob * validf
            ones    = ones    * validf

        intersection = torch.zeros(B, C, device=probs.device, dtype=probs.dtype)
        count        = torch.zeros(B, C, device=probs.device, dtype=probs.dtype)
        intersection.scatter_add_(1, tgt_flat, gt_prob)
        count.scatter_add_(1, tgt_flat, ones)

        if self.ignore_index is not None:
            count       [:, self.ignore_index] = 0
            intersection[:, self.ignore_index] = 0

        probs_sum = probs_flat.sum(2)                                   # (B,C)
        denom     = probs_sum + count
        dice      = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        loss      = 1.0 - dice                                          # (B,C)

        present_gt = count > 0
        if not present_gt.any():
            return loss.new_tensor(0.0)

        pf        = present_gt.float()
        per_image = (loss * pf).sum(1) / pf.sum(1).clamp(min=1)        # (B,)
        return per_image.mean()


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

    def _present_class_reduce(
        self,
        pixel_loss: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reduce focal loss by averaging equally over GT-present classes per image.
        Vectorized via scatter_add: replaces O(B*C) Python-dispatched GPU ops
        with ~6 fused kernel calls, recovering ~10-15s/epoch vs the loop version.
        """
        B, H, W = pixel_loss.shape
        C = self.num_classes

        flat_loss = pixel_loss.reshape(B, -1)   # (B, N)
        flat_tgt  = target.reshape(B, -1)        # (B, N)

        if self.ignore_index >= 0:
            valid     = flat_tgt != self.ignore_index
            flat_loss = flat_loss * valid.float()
            flat_tgt  = flat_tgt.clamp(min=0)

        # Accumulate per-(image, class) loss sum and pixel count
        loss_sum = torch.zeros(B, C, device=pixel_loss.device, dtype=pixel_loss.dtype)
        count    = torch.zeros(B, C, device=pixel_loss.device, dtype=pixel_loss.dtype)
        loss_sum.scatter_add_(1, flat_tgt, flat_loss)
        count.scatter_add_(1, flat_tgt, torch.ones_like(flat_loss))

        if self.ignore_index >= 0:
            count   [:, self.ignore_index] = 0
            loss_sum[:, self.ignore_index] = 0

        present   = count > 0                                          # (B, C)
        class_mean = loss_sum / count.clamp(min=1)                     # (B, C)
        pf        = present.float()
        per_image = (class_mean * pf).sum(1) / pf.sum(1).clamp(min=1) # (B,)
        return per_image.mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self._pixel_loss(logits, target)
        return self._present_class_reduce(loss, target)


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
        pixel_loss = self.focal._pixel_loss(logits, target)
        focal_loss = self.focal._present_class_reduce(pixel_loss, target)

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
