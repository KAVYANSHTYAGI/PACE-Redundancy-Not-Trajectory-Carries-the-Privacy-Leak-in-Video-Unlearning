"""Training / evaluation for the VKITTI2 reference models, mirroring
seg/engine.py exactly except parameterized by CARCLS/NUM_CLASSES instead of
PERSON/VSPW's 124 classes. confusion_matrix/iou_from_cm are dataset-agnostic
already (seg/engine.py) and reused directly.
"""
import numpy as np
import torch
import torch.nn as nn

from seg.data import IGNORE
from seg.engine import confusion_matrix, iou_from_cm
from seg.model import seg_forward
from seg.vkitti_data import CARCLS


def seg_metrics(model, loader, device="cuda", num_classes=None):
    from seg.vkitti_data import NUM_CLASSES as VK_NUM_CLASSES
    nc = num_classes or VK_NUM_CLASSES
    cm = confusion_matrix(model, loader, device, num_classes=nc)
    iou, present = iou_from_cm(cm)
    car_iou = float(iou[CARCLS]) if present[CARCLS] else float("nan")
    retain_mask = present.copy(); retain_mask[CARCLS] = False
    retain_miou = float(np.mean(iou[retain_mask])) if retain_mask.any() else float("nan")
    miou = float(np.mean(iou[present]))
    return {"car_IoU": 100 * car_iou,
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
            msg += f" | car_IoU {m['car_IoU']:.1f} retain_mIoU {m['retain_mIoU']:.1f} mIoU {m['mIoU']:.1f}"
        print(msg, flush=True)
    return model
