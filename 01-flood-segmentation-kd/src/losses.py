"""Losses.

Two groups:
  * segmentation losses  - used by every run (baseline and proposed)
  * distillation losses  - used only by the proposed framework

All of them respect the -1 ignore mask.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Pixels marked -1 have no label, so they must not affect any loss.
IGNORE_INDEX = -1


# ------------------------------------------------------------ segmentation
class DiceLoss(nn.Module):
    """Soft Dice on the water class, computed only over valid pixels."""

    def __init__(self, positive: int = 1, eps: float = 1e-6):
        super().__init__()
        self.positive = positive      # class 1 is water
        self.eps = eps                # stops division by zero

    def forward(self, logits, target):
        valid = (target != IGNORE_INDEX).float()      # 1 where a label exists
        # softmax turns raw scores into probabilities; take the water one.
        prob = logits.softmax(1)[:, self.positive]
        tgt = (target == self.positive).float()       # 1 where water really is
        # Multiplying by valid zeroes out the unlabelled pixels.
        prob, tgt = prob * valid, tgt * valid
        # How much predicted water sits on top of real water.
        inter = (prob * tgt).sum(dim=(1, 2))
        # How much water there is in the two put together.
        denom = prob.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))
        # Perfect overlap gives 1, so subtract from 1 to make it a loss.
        # This measure ignores how much land there is, which is why it survives
        # the imbalance between water and land.
        return (1 - (2 * inter + self.eps) / (denom + self.eps)).mean()


class SegLoss(nn.Module):
    """Cross-entropy + Dice. class_weights helps because water is the rare class."""

    def __init__(self, dice_weight: float = 0.5, class_weights=None):
        super().__init__()
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        # CrossEntropyLoss handles the ignore value itself, and the weights make
        # a mistake on water count more than a mistake on land.
        self.ce = nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE_INDEX)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        # A chip that is entirely cloud has no usable pixel at all. Averaging
        # over nothing gives an undefined number, so return a zero that still
        # carries the gradient machinery.
        if not (target != IGNORE_INDEX).any():
            return logits.sum() * 0.0
        # Per-pixel correctness plus overlap of the water shape.
        return self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


# ------------------------------------------------------------ distillation
class ResponseKD(nn.Module):
    """Classic Hinton-style KD on the output logits, masked to valid pixels.

    This is the 'pixel-wise distillation' baseline that Liu et al. (2019) showed
    is NOT enough on its own for segmentation. Keep it as the control.
    """

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.t = temperature          # higher means a softer, more graded answer

    def forward(self, student_logits, teacher_logits, target=None):
        # Dividing by the temperature flattens the answer, so instead of a firm
        # yes or no the student sees how sure the teacher was.
        s = F.log_softmax(student_logits / self.t, dim=1)
        t = F.softmax(teacher_logits / self.t, dim=1)
        # KL divergence measures how far apart two opinions are.
        kl = F.kl_div(s, t, reduction='none').sum(1)          # (B, H, W)
        if target is not None:
            valid = (target != IGNORE_INDEX).float()
            kl = kl * valid                                   # ignore no-data
            # Average over the valid pixels only. The T squared puts the
            # gradients back to their normal size after the softening.
            return (kl.sum() / valid.sum().clamp(min=1)) * (self.t ** 2)
        return kl.mean() * (self.t ** 2)


class PairwiseKD(nn.Module):
    """Pair-wise distillation from Liu et al. (2019).

    Matches the pixel-to-pixel similarity map of teacher and student instead of
    their per-pixel probabilities, so spatial structure is transferred.
    Features are pooled first because a full 512x512 similarity matrix will not
    fit in memory.
    """

    def __init__(self, pool: int = 16):
        super().__init__()
        self.pool = pool              # shrink the feature map to 16 by 16 first

    @staticmethod
    def _similarity(feat):
        b, c, h, w = feat.shape
        # Lay the grid out as a list of regions, each with its own feature vector.
        f = feat.reshape(b, c, h * w)
        # normalize makes every vector the same length, so the comparison below
        # is about direction only, not how strong the response was.
        f = F.normalize(f, dim=1)
        # Multiplying the list by itself gives every region compared with every
        # other region: the network's own sense of what looks like what.
        return torch.bmm(f.transpose(1, 2), f)                # (B, HW, HW)

    def forward(self, student_feat, teacher_feat):
        # 16 by 16 gives 256 regions and so 65,536 pairs. The full 512 by 512
        # map would be 68 billion, which is why the pooling is not optional.
        s = F.adaptive_avg_pool2d(student_feat, self.pool)
        t = F.adaptive_avg_pool2d(teacher_feat, self.pool)
        # Punish the student wherever its sense of resemblance differs.
        return F.mse_loss(self._similarity(s), self._similarity(t))


class ConfidenceGate(nn.Module):
    """Confidence gating, following the idea in Ma et al. (2026).

    An optical teacher observing a partly clouded scene is not reliable
    everywhere. This down-weights the distillation loss wherever the teacher is
    unsure, so bad supervision does not propagate into the student.
    """

    def __init__(self, threshold: float = 0.7):
        super().__init__()
        self.threshold = threshold

    def forward(self, teacher_logits):
        # The highest probability is how sure the teacher is about that pixel.
        conf = teacher_logits.softmax(1).max(1).values           # (B, H, W)
        # 1 where the teacher is confident enough to be worth copying, 0 elsewhere.
        return (conf >= self.threshold).float()


class DistillLoss(nn.Module):
    """Total objective for the proposed framework.

        L = L_seg + alpha * L_response + beta * L_pairwise

    Set alpha/beta to 0 in the config to ablate a component.
    """

    def __init__(self, seg_loss, alpha=1.0, beta=1.0, temperature=4.0,
                 gate_threshold=None, pool=16):
        super().__init__()
        self.seg = seg_loss                       # against the real labels
        self.response = ResponseKD(temperature)   # against the teacher's answer
        self.pairwise = PairwiseKD(pool)          # against the teacher's shapes
        self.alpha, self.beta = alpha, beta       # how much each one counts
        # No threshold means no gate at all.
        self.gate = ConfidenceGate(gate_threshold) if gate_threshold else None

    def forward(self, s_logits, t_logits, target, s_feat=None, t_feat=None):
        # Always start with the loss against the hand-drawn labels. It is what
        # keeps the student tied to reality rather than only to the teacher.
        out = {'seg': self.seg(s_logits, target)}

        if self.alpha > 0:            # alpha 0 switches this term off entirely
            if self.gate is not None:
                mask = self.gate(t_logits)
                # Where the teacher is unsure, mark the pixel as no-data so the
                # response loss skips it. Reusing the ignore value means the
                # masking logic lives in one place.
                masked_target = torch.where(mask.bool(), target,
                                            torch.full_like(target, IGNORE_INDEX))
                out['response'] = self.alpha * self.response(s_logits, t_logits, masked_target)
            else:
                out['response'] = self.alpha * self.response(s_logits, t_logits, target)

        # The pair-wise term needs the bottleneck features from both networks.
        if self.beta > 0 and s_feat is not None and t_feat is not None:
            out['pairwise'] = self.beta * self.pairwise(s_feat, t_feat)

        # Add up whatever terms are switched on. This one number is what
        # training actually tries to make smaller.
        out['total'] = sum(v for k, v in out.items() if k != 'total')
        return out
