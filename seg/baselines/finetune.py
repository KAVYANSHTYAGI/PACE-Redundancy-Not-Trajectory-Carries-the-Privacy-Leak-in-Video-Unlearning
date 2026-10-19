"""FT (Finetune): faithful port of SalUn's FT.

Source: vendor/SalUn/Classification/unlearn/FT.py (`FT_iter`).
Original: finetune the model on the RETAIN set only with cross-entropy; the target
class is forgotten by catastrophic forgetting. Default SalUn hyper-params:
unlearn_epochs=10, unlearn_lr=0.01, SGD momentum 0.9, weight_decay 5e-4.

Segmentation adaptation: cross-entropy on the RETAIN pixels (non-person, non-ignore)
of every image; forget pixels receive no supervision. Nothing else changes.
"""
import torch.nn as nn

from seg.model import seg_forward
from ._common import clone_model, masked_ce, pixel_masks, sgd


def run(model, loader, device="cuda", epochs=10, lr=0.01, **_):
    m = clone_model(model).to(device).train()
    opt = sgd(m.parameters(), lr)
    for _e in range(epochs):
        for img, orig, _merged in loader:
            img, orig = img.to(device), orig.to(device)
            _, retain = pixel_masks(orig)
            loss = masked_ce(seg_forward(m, img), orig, retain)
            opt.zero_grad(); loss.backward(); opt.step()
    return m
