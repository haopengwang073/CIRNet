"""Per-image metrics; bad-pixel rates and D1 are percentages."""

import torch


def image_metrics(prediction, ground_truth, mask):
    if (
        prediction.ndim != 2
        or prediction.shape != ground_truth.shape
        or mask.shape != prediction.shape
    ):
        raise ValueError(
            "prediction, ground truth, and mask must be equal-sized 2D arrays"
        )
    mask = mask.bool()
    count = mask.sum()
    if count == 0:
        raise ValueError("no valid evaluation pixels")
    if (
        not torch.isfinite(prediction).all()
        or not torch.isfinite(ground_truth[mask]).all()
    ):
        raise ValueError("non-finite prediction or valid ground truth")
    error = (prediction - ground_truth).abs()
    d1 = (
        ((error > 3) & (error / ground_truth.abs() > 0.05) & mask).sum().float()
        / count.float()
        * 100
    )
    epe = torch.where(mask, error, torch.zeros_like(error)).sum() / count
    metrics = {"EPE": epe.item(), "D1": d1.item()}
    for threshold in (1, 2, 3):
        bad = ((error > threshold) & mask).sum().float() / count.float() * 100
        metrics[f"Bad{threshold}"] = bad.item()
    return metrics
