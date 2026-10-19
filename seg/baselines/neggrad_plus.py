"""NegGrad+: faithful port of the SCRUB paper's NegGrad+ baseline.

Source: Kurmanji et al., "Towards Unbounded Machine Unlearning" (NeurIPS 2023),
vendor/SCRUB/thirdparty/repdistiller/helper/loops.py (`train_negrad`).
Original (train_negrad): per batch, take a retain batch and a forget batch and
optimise  loss = beta * CE(retain) - (1 - beta) * CE(forget)  jointly (descend on
retain, ascend on forget). Their default beta ~ 0.999.

Segmentation adaptation: within each image, CE on retain pixels minus CE on forget
pixels, both from the same forward pass.
"""
from seg.model import seg_forward
from ._common import clone_model, masked_ce, pixel_masks, sgd


def run(model, loader, device="cuda", epochs=5, lr=0.01, beta=0.999, **_):
    m = clone_model(model).to(device).train()
    opt = sgd(m.parameters(), lr)
    for _e in range(epochs):
        for img, orig, _merged in loader:
            img, orig = img.to(device), orig.to(device)
            forget, retain = pixel_masks(orig)
            logits = seg_forward(m, img)
            loss = beta * masked_ce(logits, orig, retain) \
                - (1 - beta) * masked_ce(logits, orig, forget)
            opt.zero_grad(); loss.backward(); opt.step()
    return m
