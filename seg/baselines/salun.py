"""SalUn (Saliency Unlearning): faithful port of Fan et al., ICLR 2024.

Source: vendor/SalUn/Classification/generate_mask.py (`save_gradient_ratio`) + RL.
Original mask: accumulate the absolute gradient of the forget cross-entropy w.r.t.
each weight, GLOBALLY rank all weights by that magnitude, and set a hard mask = 1 for
the top-`threshold` fraction (their released mask uses threshold 0.5). Unlearning then
runs Random-Label finetuning updating ONLY the masked (salient) weights.

Segmentation adaptation: the forget gradient is the CE on forget (person) pixels; the
global top-fraction masking and the masked RL update are identical to the source.
"""
import torch

from seg.model import seg_forward
from ._common import clone_model, masked_ce, pixel_masks
from .random_label import run as rl_run


def saliency_mask(model, loader, device, threshold=0.5, max_batches=50):
    m = clone_model(model).to(device).train()
    grads = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
    for i, (img, orig, _merged) in enumerate(loader):
        if i >= max_batches:
            break
        img, orig = img.to(device), orig.to(device)
        forget, _ = pixel_masks(orig)
        m.zero_grad()
        masked_ce(seg_forward(m, img), orig, forget).backward()
        for n, p in m.named_parameters():
            if p.grad is not None:
                grads[n] += p.grad.data                          # accumulate (then abs)
    for n in grads:
        grads[n] = torch.abs_(grads[n])

    # global rank: top-`threshold` fraction of weights by |grad| -> mask 1 (SalUn)
    all_elements = -torch.cat([g.flatten() for g in grads.values()])
    positions = torch.argsort(all_elements)
    ranks = torch.argsort(positions)
    thr_index = int(len(all_elements) * threshold)
    flat_mask = torch.zeros_like(ranks)
    flat_mask[ranks < thr_index] = 1
    mask, off = {}, 0
    for n, g in grads.items():
        k = g.numel()
        mask[n] = flat_mask[off:off + k].reshape(g.shape).float()
        off += k
    return mask


def run(model, loader, device="cuda", epochs=10, lr=0.013, threshold=0.5, **_):
    mask = saliency_mask(model, loader, device, threshold=threshold)
    return rl_run(model, loader, device=device, epochs=epochs, lr=lr, mask=mask)
