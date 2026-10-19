"""Shared utilities for the segmentation-unlearning baselines.

Every baseline in this package is a FAITHFUL port of a published method's official
code / equations (source cited in each module's docstring), adapted to dense
prediction by the same rule everywhere:

  * classification loss on a *sample*  ->  loss masked to that sample's pixels
  * forget set  ->  forget pixels  F = {i : y_i == PERSON}
  * retain set  ->  retain pixels  R = {i : y_i != PERSON and y_i != IGNORE}

so a single image contributes both forget and retain pixels (the segmentation-
specific entanglement). This module holds the primitives the ports build on; it does
NOT invent any method logic of its own.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from seg.data import IGNORE, NUM_CLASSES, PERSON
from seg.model import seg_forward


def clone_model(model):
    return copy.deepcopy(model)


def pixel_masks(orig):
    """forget = person pixels; retain = valid non-person pixels."""
    forget = orig == PERSON
    retain = (orig != PERSON) & (orig != IGNORE)
    return forget, retain


def masked_ce(logits, target, mask):
    """Mean cross-entropy over pixels where `mask` is True. `target` is clamped so
    that negative/ignore placeholders under the mask don't index out of range."""
    if mask.sum() == 0:
        return logits.sum() * 0.0
    loss = F.cross_entropy(logits, target.clamp(min=0), ignore_index=IGNORE,
                           reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


class DistillKL(nn.Module):
    """KL divergence for distillation, exactly as in RepDistiller / SCRUB
    (thirdparty/repdistiller/distiller_zoo/KD.py): T^2 * KL(log_softmax(s/T),
    softmax(t/T)). Here evaluated per-pixel and averaged over a mask."""

    def __init__(self, T=4.0):
        super().__init__()
        self.T = T

    def forward(self, logit_s, logit_t, mask=None):
        p_s = F.log_softmax(logit_s / self.T, dim=1)
        p_t = F.softmax(logit_t / self.T, dim=1)
        kl = F.kl_div(p_s, p_t, reduction="none").sum(1) * (self.T ** 2)   # [N,H,W]
        if mask is None:
            return kl.mean()
        if mask.sum() == 0:
            return logit_s.sum() * 0.0
        return (kl * mask).sum() / mask.sum().clamp(min=1)


def sgd(params, lr, momentum=0.9, weight_decay=5e-4):
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
