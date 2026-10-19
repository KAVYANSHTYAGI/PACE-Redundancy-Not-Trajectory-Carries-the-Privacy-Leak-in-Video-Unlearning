"""RL (Random Label): faithful port of SalUn's RL.

Source: vendor/SalUn/Classification/unlearn/RL.py (`RL`).
Original: relabel every forget sample with a RANDOM class
(`np.random.randint(0, num_classes)`), concatenate with the retain set, and finetune
with cross-entropy on the union. Default SalUn: unlearn_epochs=10, unlearn_lr=0.013.
(SalUn = RL restricted to a saliency mask; see salun.py.)

Segmentation adaptation: each forget (person) pixel gets an i.i.d. random class label
(resampled to differ from PERSON); cross-entropy is applied on forget pixels (random
targets) and retain pixels (true targets) from the same forward pass.
"""
import torch

from seg.data import NUM_CLASSES, PERSON
from seg.model import seg_forward
from ._common import clone_model, masked_ce, pixel_masks, sgd


def run(model, loader, device="cuda", epochs=10, lr=0.013, mask=None, **_):
    m = clone_model(model).to(device).train()
    opt = sgd(m.parameters(), lr)
    for _e in range(epochs):
        for img, orig, _merged in loader:
            img, orig = img.to(device), orig.to(device)
            forget, retain = pixel_masks(orig)
            rand = torch.randint(0, NUM_CLASSES, orig.shape, device=device)
            rand[rand == PERSON] = (PERSON + 1) % NUM_CLASSES     # random != person
            target = torch.where(forget, rand, orig)
            logits = seg_forward(m, img)
            loss = masked_ce(logits, target, forget) + masked_ce(logits, orig, retain)
            opt.zero_grad(); loss.backward()
            if mask is not None:                                 # SalUn: restrict to salient params
                for n, p in m.named_parameters():
                    if p.grad is not None:
                        p.grad *= mask[n]
            opt.step()
    return m
