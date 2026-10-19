"""Pixel-level unlearning baselines for VSPW person-forgetting.

Faithful port of project1/seg/methods.py -- the classification-unlearning methods
adapted to dense prediction. They are frame-i.i.d. (consume VSPWUnlearn's
(img, orig, merged) tuples exactly as project1 consumed ADE20KUnlearn's), so the
logic is unchanged; only the dataset behind `loader` differs. These are the
per-frame comparison baselines the temporal results are measured against.

The temporal method (ADVR-T) lives in candidates.py; these baselines are
deliberately per-frame so the paper can show that per-frame unlearning -- even
when it succeeds per-frame -- leaves trajectory-level leakage that only a
temporal method removes.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from seg.data import IGNORE, NUM_CLASSES, PERSON
from seg.model import seg_forward


def _clone(model):
    return copy.deepcopy(model)


def _ce_none(logits, target):
    return F.cross_entropy(logits, target, ignore_index=IGNORE, reduction="none")


def _masked_ce(logits, target, mask):
    """Mean CE over pixels where mask is True (mask already excludes ignore)."""
    if mask.sum() == 0:
        return logits.sum() * 0.0
    loss = _ce_none(logits, target.clamp(min=0))
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def _masks(orig):
    forget = orig == PERSON
    retain = (orig != PERSON) & (orig != IGNORE)
    return forget, retain


# --------------------------------------------------------------------------- #
def ft(model, loader, device="cuda", epochs=3, lr=1e-3, **_):
    """Finetune toward the merged target (person -> neighbor)."""
    m = _clone(model).to(device).train()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(ignore_index=IGNORE)
    for _e in range(epochs):
        for img, _orig, merged in loader:
            img, merged = img.to(device), merged.to(device)
            opt.zero_grad(); crit(seg_forward(m, img), merged).backward(); opt.step()
    return m


def ga(model, loader, device="cuda", epochs=2, lr=1e-4, **_):
    """Gradient ascent on person pixels only (maximize CE there)."""
    m = _clone(model).to(device).train()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    for _e in range(epochs):
        for img, orig, _merged in loader:
            img, orig = img.to(device), orig.to(device)
            forget, _ = _masks(orig)
            logits = seg_forward(m, img)
            loss = -_masked_ce(logits, orig, forget)
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def neggrad_plus(model, loader, device="cuda", epochs=2, lr=1e-3, alpha=0.95, **_):
    """Descend on retain pixels (orig labels), ascend on person pixels."""
    m = _clone(model).to(device).train()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    for _e in range(epochs):
        for img, orig, _merged in loader:
            img, orig = img.to(device), orig.to(device)
            forget, retain = _masks(orig)
            logits = seg_forward(m, img)
            loss = alpha * _masked_ce(logits, orig, retain) \
                - (1 - alpha) * _masked_ce(logits, orig, forget)
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def rl(model, loader, device="cuda", epochs=3, lr=1e-3, mask=None, **_):
    """Random per-pixel labels on person pixels (!= person), CE on retain pixels."""
    m = _clone(model).to(device).train()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    for _e in range(epochs):
        for img, orig, _merged in loader:
            img, orig = img.to(device), orig.to(device)
            forget, retain = _masks(orig)
            rand = torch.randint(0, NUM_CLASSES, orig.shape, device=device)
            rand[rand == PERSON] = (PERSON + 1) % NUM_CLASSES
            target = torch.where(forget, rand, orig)
            logits = seg_forward(m, img)
            loss = _masked_ce(logits, target, forget) + _masked_ce(logits, orig, retain)
            opt.zero_grad(); loss.backward()
            if mask is not None:
                for n, p in m.named_parameters():
                    if p.grad is not None:
                        p.grad *= mask[n]
            opt.step()
    return m


def _saliency_mask(model, loader, device, threshold=0.5, max_batches=50):
    m = _clone(model).to(device).train()
    grads = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
    for i, (img, orig, _merged) in enumerate(loader):
        if i >= max_batches:
            break
        img, orig = img.to(device), orig.to(device)
        forget, _ = _masks(orig)
        m.zero_grad()
        _masked_ce(seg_forward(m, img), orig, forget).backward()
        for n, p in m.named_parameters():
            if p.grad is not None:
                grads[n] += p.grad.detach().abs()
    flat = torch.cat([g.flatten() for g in grads.values()])
    cutoff = torch.quantile(flat[torch.randperm(len(flat))[:1_000_000]], threshold)
    return {n: (g >= cutoff).float() for n, g in grads.items()}


def salun(model, loader, device="cuda", epochs=3, lr=1e-3, threshold=0.5, **_):
    mask = _saliency_mask(model, loader, device, threshold=threshold)
    return rl(model, loader, device=device, epochs=epochs, lr=lr, mask=mask)


def ssd(model, loader, device="cuda", dampening_constant=1.0,
        selection_weighting=1.0, exponent=1.0, lower_bound=1.0, **_):
    """SSD with forget importance = person-pixel loss, full importance = all pixels."""
    m = _clone(model).to(device)

    def importance(use_forget_mask):
        imp = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
        m.eval()
        nb = 0
        for img, orig, _merged in loader:
            img, orig = img.to(device), orig.to(device)
            forget, retain = _masks(orig)
            mask = forget if use_forget_mask else (forget | retain)
            m.zero_grad()
            _masked_ce(seg_forward(m, img), orig, mask).backward()
            for n, p in m.named_parameters():
                if p.grad is not None:
                    imp[n] += p.grad.detach().pow(2)
            nb += 1
        for n in imp:
            imp[n] /= max(nb, 1)
        return imp

    forget_imp = importance(True)
    full_imp = importance(False)
    with torch.no_grad():
        for n, p in m.named_parameters():
            oimp, fimp = full_imp[n], forget_imp[n]
            loc = torch.where(fimp > oimp.mul(selection_weighting))
            weight = (oimp.mul(dampening_constant).div(fimp + 1e-12)).pow(exponent)
            upd = weight[loc]
            upd[upd > lower_bound] = lower_bound
            p[loc] = p[loc].mul(upd)
    return m


def scrub(model, loader, device="cuda", epochs=3, lr=5e-4, kd_T=4.0,
          alpha=0.5, gamma=1.0, msteps=1, **_):
    student = _clone(model).to(device).train()
    teacher = _clone(model).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    def kl_px(s_logits, t_logits, mask):
        if mask.sum() == 0:
            return s_logits.sum() * 0.0
        s = F.log_softmax(s_logits / kd_T, 1)
        t = F.softmax(t_logits / kd_T, 1)
        kld = F.kl_div(s, t, reduction="none").sum(1)        # [N,H,W]
        return (kld * mask).sum() / mask.sum().clamp(min=1) * (kd_T ** 2)

    for e in range(epochs):
        for img, orig, _merged in loader:
            img, orig = img.to(device), orig.to(device)
            forget, retain = _masks(orig)
            s_log = seg_forward(student, img)
            with torch.no_grad():
                t_log = seg_forward(teacher, img)
            loss = gamma * _masked_ce(s_log, orig, retain) \
                + alpha * kl_px(s_log, t_log, retain)
            if e < msteps:
                loss = loss - kl_px(s_log, t_log, forget)     # push away on person
            opt.zero_grad(); loss.backward(); opt.step()
    return student


METHODS = {"ft": ft, "ga": ga, "neggrad+": neggrad_plus, "rl": rl,
           "salun": salun, "ssd": ssd, "scrub": scrub}


def run_method(name, model, loader, device="cuda", **kw):
    if name not in METHODS:
        raise ValueError(f"unknown method {name!r}; have {list(METHODS)}")
    return METHODS[name](model, loader, device=device, **kw)
