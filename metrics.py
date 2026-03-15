import torch


@torch.no_grad()
def confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
):
    """
    pred, target: (B,H,W) long
    """
    pred = pred.view(-1)
    target = target.view(-1)

    if ignore_index is not None:
        mask = target != ignore_index
        pred = pred[mask]
        target = target[mask]

    k = (target >= 0) & (target < num_classes)
    pred = pred[k]
    target = target[k]

    idx = target * num_classes + pred
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )
    return cm


@torch.no_grad()
def f1_macro_from_cm(cm: torch.Tensor, eps: float = 1e-8):
    """
    cm: (C,C) where rows=gt, cols=pred
    returns macro-F1 over classes
    """
    tp = torch.diag(cm).float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1.mean().item(), f1.cpu()  # macro, per-class tensor


@torch.no_grad()
def fbeta_present_gt_from_cm(
    cm: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-8,
):
    """
    cm: (C,C) where rows=gt, cols=pred
    returns mean F-beta over classes present in GT only

    This matches the user's numpy implementation:
      - average only over class ids appearing in ground truth
      - include FP/FN for those classes
      - ignore classes absent from GT when averaging
    """
    tp = torch.diag(cm).float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp

    beta2 = beta**2
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall + eps)

    present_gt = cm.sum(1) > 0
    if present_gt.any():
        return fbeta[present_gt].mean().item(), fbeta.cpu(), present_gt.cpu()
    return 0.0, fbeta.cpu(), present_gt.cpu()
