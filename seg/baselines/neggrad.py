"""GA / Negative Gradient: faithful port of SalUn's GA.

Source: vendor/SalUn/Classification/unlearn/GA.py (`GA`).
Original: gradient ASCENT on the forget set, i.e. minimise the negated
cross-entropy  loss = -CE(f(x_forget), y_forget)  with SGD. Default SalUn:
unlearn_epochs=5, unlearn_lr=1e-4 (small, since ascent is unstable).

Segmentation adaptation: ascend on the forget (person) pixels' cross-entropy.
"""
import torch

from seg.model import seg_forward
from ._common import clone_model, masked_ce, pixel_masks, sgd


def run(model, loader, device="cuda", epochs=5, lr=1e-4, **_):
    m = clone_model(model).to(device).train()
    opt = sgd(m.parameters(), lr)
    for _e in range(epochs):
        for img, orig, _merged in loader:
            img, orig = img.to(device), orig.to(device)
            forget, _ = pixel_masks(orig)
            if forget.sum() == 0:
                continue
            loss = -masked_ce(seg_forward(m, img), orig, forget)   # ascent
            opt.zero_grad(); loss.backward()
            # unbounded ascent (loss -> -inf) diverges to NaN without a cap,
            # same failure mode found and fixed in scrub.py's MAX step
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=5.0)
            opt.step()
    return m
