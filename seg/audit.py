"""Privacy/leakage metrics + relearning audit for VSPW person-unlearning.

Per-frame metrics (ported from project1): residual_person, blend_fail,
silhouette_IoU -- the "did the label go / did the shape blend / can a naive
attacker recover the frame mask" trio.

Temporal metric (new): trajectory_leakage -- runs the trained temporal attacker
(seg/attack) over clips and reports its trajectory IoU/AUC plus the
temporal-consistency of the recovered mask. This is the quantity the ADVR-T
method is designed to drive to the oracle floor; the per-frame metrics can look
clean (label gone, shape blended) while trajectory_leakage is still high -- that
gap is the paper.
"""
import copy

import numpy as np
import torch

from seg.data import IGNORE, PERSON
from seg.engine import seg_metrics
from seg.methods import _masked_ce
from seg.model import seg_forward


@torch.no_grad()
def leakage_metrics(model, loader, device="cuda"):
    """loader yields (img, orig, merged) single frames. Aggregates over person frames."""
    model.eval()
    res_person = res_total = 0
    blend_fail = blend_total = 0
    sil_inter = sil_union = 0
    for img, orig, merged in loader:
        img, orig, merged = img.to(device), orig.to(device), merged.to(device)
        pred = seg_forward(model, img).argmax(1)
        M = orig == PERSON
        if M.sum() == 0:
            continue
        res_person += (pred[M] == PERSON).sum().item(); res_total += M.sum().item()
        blend_fail += (pred[M] != merged[M]).sum().item(); blend_total += M.sum().item()
        anomaly = (pred != merged) & (orig != IGNORE)
        sil_inter += (anomaly & M).sum().item(); sil_union += (anomaly | M).sum().item()
    return {
        "residual_person": 100 * res_person / max(res_total, 1),
        "blend_fail": 100 * blend_fail / max(blend_total, 1),
        "silhouette_IoU": 100 * sil_inter / max(sil_union, 1),
    }


def relearn_audit(unlearned_model, loader, device="cuda", steps=50, lr=1e-3,
                  val_loader=None):
    """Few-shot finetune on person pixels; report recovered person_IoU.
    `loader` yields (img, orig, merged) single frames; `val_loader` yields (img, y)."""
    m = copy.deepcopy(unlearned_model).to(device).train()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
    before = seg_metrics(m, val_loader, device)["person_IoU"] if val_loader else None
    it = iter(loader); done = 0
    while done < steps:
        try:
            img, orig, _merged = next(it)
        except StopIteration:
            it = iter(loader); continue
        img, orig = img.to(device), orig.to(device)
        forget = orig == PERSON
        m.train()
        loss = _masked_ce(seg_forward(m, img), orig, forget)
        opt.zero_grad(); loss.backward(); opt.step(); done += 1
    after = seg_metrics(m, val_loader, device)["person_IoU"] if val_loader else None
    out = {"person_IoU_before": before, "person_IoU_after": after}
    if before is not None:
        out["recover_pts"] = round(after - before, 1)
    return out


def trajectory_leakage(model, clip_train_loader, clip_val_loader, device="cuda",
                       transform=None, kind="conv3d", feat=False, out_size=128,
                       epochs=2, max_iters=400):
    """Train a temporal attacker against `model` on clip_train_loader and evaluate
    trajectory reconstruction on clip_val_loader. Direct temporal analog of
    training+evaluating the per-frame silhouette attacker. Returns the temporal
    attacker's trajectory IoU/AUC and traj_consistency."""
    from seg.attack import train_temporal_attacker, eval_temporal_attacker
    D = train_temporal_attacker(model, clip_train_loader, device, transform=transform,
                                epochs=epochs, max_iters=max_iters, kind=kind,
                                feat=feat, out_size=out_size)
    return eval_temporal_attacker(D, model, clip_val_loader, device,
                                  transform=transform, feat=feat, out_size=out_size)
