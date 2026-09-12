"""Segmentation metrics that ignore the -1 no-data pixels.

Every number the project reports comes from here, so that the ignore mask is
applied in exactly one place and cannot be forgotten in some other script.
"""
from __future__ import annotations

import torch

# The label value that means "no information here", usually because cloud hid
# the scene when the mask was drawn. It must never count towards a score.
IGNORE_INDEX = -1


class SegMetrics:
    """Accumulates a confusion matrix over batches, then reports water-class scores."""

    def __init__(self, num_classes: int = 2, device: str = 'cpu'):
        self.num_classes = num_classes    # two: water and not water
        self.device = device              # keep the counts where the model is
        self.reset()

    def reset(self) -> None:
        # A 2 by 2 table of counts. Row is the true class, column is what the
        # model predicted, so cm[1][0] means "real water called land".
        self.cm = torch.zeros(self.num_classes, self.num_classes,
                              dtype=torch.long, device=self.device)

    @torch.no_grad()      # no learning happens here, so do not track gradients
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """logits (B, C, H, W) raw scores; target (B, H, W) with -1/0/1."""
        # The model gives a score per class; the highest one is its answer.
        pred = logits.argmax(1)
        valid = target != IGNORE_INDEX          # <- the whole point
        # Flatten to a long list of pixels, keeping only the usable ones.
        p = pred[valid].reshape(-1)
        t = target[valid].reshape(-1)
        # Turn each (true, predicted) pair into a single number, so 0 means
        # land called land, 1 land called water, 2 water called land, 3 both water.
        idx = t * self.num_classes + p
        # Count how many pixels fall into each of those four cases.
        binc = torch.bincount(idx, minlength=self.num_classes ** 2)
        # Fold the counts back into the 2 by 2 table and add to the running total.
        self.cm += binc.reshape(self.num_classes, self.num_classes).to(self.cm.device)

    def compute(self, positive: int = 1) -> dict:
        """positive=1 means the water class."""
        cm = self.cm.float()                    # divide needs decimals
        tp = cm[positive, positive]             # water correctly found
        fn = cm[positive].sum() - tp            # water missed
        fp = cm[:, positive].sum() - tp         # land wrongly called water
        tn = cm.sum() - tp - fn - fp            # land correctly left alone
        eps = 1e-9                              # stops division by zero

        # Overlap between what was predicted and what is really there.
        iou = tp / (tp + fp + fn + eps)
        # Of everything called water, how much really was.
        precision = tp / (tp + fp + eps)
        # Of all the real water, how much was found.
        recall = tp / (tp + fn + eps)
        # One number combining the two, low unless both are decent.
        f1 = 2 * precision * recall / (precision + recall + eps)
        # Every pixel judged correctly, water or land. Misleading here, because
        # water is rare, but reported for completeness.
        acc = (tp + tn) / (cm.sum() + eps)

        # mean IoU over both classes, the metric most papers report
        ious = []
        for c in range(self.num_classes):
            t_ = cm[c, c]                       # correct for this class
            f_n = cm[c].sum() - t_              # missed
            f_p = cm[:, c].sum() - t_           # wrongly claimed
            ious.append((t_ / (t_ + f_p + f_n + eps)).item())

        # .item() pulls a plain Python number out of a one-value tensor.
        return {
            'iou_water': iou.item(),
            'miou': sum(ious) / len(ious),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'accuracy': acc.item(),
            'n_valid_px': int(cm.sum().item()),   # how many pixels were judged
        }

    def __str__(self) -> str:
        # What gets shown when this object is printed.
        m = self.compute()
        return (f"IoU(water) {m['iou_water']:.4f} | mIoU {m['miou']:.4f} | "
                f"F1 {m['f1']:.4f} | P {m['precision']:.4f} | R {m['recall']:.4f} | "
                f"Acc {m['accuracy']:.4f}")
