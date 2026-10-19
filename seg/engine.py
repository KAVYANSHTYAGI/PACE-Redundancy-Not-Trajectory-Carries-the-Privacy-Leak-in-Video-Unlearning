"""Training / evaluation for the per-frame video-segmentation baseline.

Identical structure to project1/seg/engine.py: confusion-matrix-based mIoU,
person_IoU tracked separately from retain_mIoU. Frames are sampled i.i.d. from
across all videos for this baseline (no temporal batching yet) -- establishing
the non-temporal floor is step 3 of POSITIONING.md before any clip-level
training is introduced.
"""
import random

import numpy as np
import torch
import torch.nn as nn

from seg.data import IGNORE, NUM_CLASSES, PERSON
from seg.model import seg_forward


def setup_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def confusion_matrix(model, loader, device="cuda", num_classes=NUM_CLASSES):
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = seg_forward(model, x).argmax(1)
        valid = y != IGNORE
        t = y[valid]; p = pred[valid]
        idx = t * num_classes + p
        cm += torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm.cpu().numpy()


def iou_from_cm(cm):
    inter = np.diag(cm)
    union = cm.sum(1) + cm.sum(0) - inter
    iou = inter / np.maximum(union, 1)
    present = cm.sum(1) > 0
    return iou, present


def seg_metrics(model, loader, device="cuda"):
    """Returns person_IoU, retain_mIoU (mean over present classes != person),
    and overall mIoU."""
    cm = confusion_matrix(model, loader, device)
    iou, present = iou_from_cm(cm)
    person_iou = float(iou[PERSON]) if present[PERSON] else float("nan")
    retain_mask = present.copy(); retain_mask[PERSON] = False
    retain_miou = float(np.mean(iou[retain_mask])) if retain_mask.any() else float("nan")
    miou = float(np.mean(iou[present]))
    return {"person_IoU": 100 * person_iou,
            "retain_mIoU": 100 * retain_miou,
            "mIoU": 100 * miou}


def train_seg(model, loader, epochs, lr, device="cuda", weight_decay=1e-4,
              aux_weight=0.4, log_prefix="", val_loader=None, log_every=1):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.PolynomialLR(opt, total_iters=epochs * len(loader), power=0.9)
    crit = nn.CrossEntropyLoss(ignore_index=IGNORE)
    for epoch in range(epochs):
        model.train()
        running = 0.0; n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            if isinstance(out, dict) and "out" in out:
                loss = crit(out["out"], y)
                if "aux" in out:
                    loss = loss + aux_weight * crit(out["aux"], y)
            else:
                loss = crit(seg_forward(model, x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            running += loss.item() * x.size(0); n += x.size(0)
        msg = f"{log_prefix}epoch {epoch+1}/{epochs} loss {running/n:.3f} lr {opt.param_groups[0]['lr']:.5f}"
        if val_loader is not None and (epoch + 1) % log_every == 0:
            m = seg_metrics(model, val_loader, device)
            msg += f" | person_IoU {m['person_IoU']:.1f} retain_mIoU {m['retain_mIoU']:.1f} mIoU {m['mIoU']:.1f}"
        print(msg, flush=True)
    return model
