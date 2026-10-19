"""SSD (Selective Synaptic Dampening): faithful port of Foster et al., AAAI 2024.

Source: vendor/SSD/src/ssd.py (`ParameterPerturber.calc_importance` +
`modify_weight`). Exact equations reproduced:

  importance[k]        = mean over batches of (dL/d theta_k)^2            (Fisher-like)
  select where  forget_imp > selection_weighting * full_imp
  weight = ( (dampening_constant * full_imp) / forget_imp ) ^ exponent
  weight = min(weight, lower_bound)                                       (clip)
  theta_k <- theta_k * weight     (only at selected locations)

Paper/repo defaults: selection_weighting (alpha)=10, dampening_constant (lambda)=1,
exponent=1, lower_bound=1.

Segmentation adaptation: `full` importance uses CE over all valid pixels; `forget`
importance uses CE over person pixels. Importance is computed in eval mode on the
un-augmented forward (deterministic), matching SSD's use of a plain forward pass.
"""
import torch
import torch.nn as nn

from seg.model import seg_forward
from ._common import clone_model, masked_ce, pixel_masks


def _importance(model, loader, device, which):
    imp = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    model.eval()
    nb = 0
    for img, orig, _merged in loader:
        img, orig = img.to(device), orig.to(device)
        forget, retain = pixel_masks(orig)
        mask = forget if which == "forget" else (forget | retain)   # "full" = all valid
        model.zero_grad()
        masked_ce(seg_forward(model, img), orig, mask).backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                imp[n] += p.grad.data.clone().pow(2)
        nb += 1
    for n in imp:
        imp[n] /= float(max(nb, 1))
    return imp


def run(model, loader, device="cuda", selection_weighting=10.0, dampening_constant=1.0,
        exponent=1.0, lower_bound=1.0, **_):
    m = clone_model(model).to(device)
    full_imp = _importance(m, loader, device, "full")
    forget_imp = _importance(m, loader, device, "forget")
    with torch.no_grad():
        for (n, p) in m.named_parameters():
            oimp, fimp = full_imp[n], forget_imp[n]
            locations = torch.where(fimp > oimp.mul(selection_weighting))
            weight = ((oimp.mul(dampening_constant)).div(fimp + 1e-12)).pow(exponent)
            update = weight[locations]
            update[update > lower_bound] = lower_bound
            p[locations] = p[locations].mul(update)
    return m
