"""SCRUB: faithful port of Kurmanji et al., "Towards Unbounded Machine
Unlearning" (NeurIPS 2023).

Source: vendor/SCRUB/main.py + thirdparty/repdistiller/helper/loops.py
(`train_distill`, split='maximize'/'minimize'). Teacher = frozen original; student =
copy. Each epoch (for the first `msteps` epochs) does a MAX step on the forget set
(maximise the student-teacher KL divergence) followed by a MIN step on the retain set
(minimise gamma*CE(retain) + alpha*KL(student,teacher)); after `msteps`, only MIN
steps. Defaults: kd_T=4, sgda_lr~5e-4, alpha(div)=0.5, gamma(cls)=1, msteps~2.

Segmentation adaptation: the KL and CE are evaluated per-pixel and averaged over the
forget (max step) or retain (min step) pixel masks.
"""
import torch

from seg.model import seg_forward
from ._common import DistillKL, clone_model, masked_ce, pixel_masks, sgd


def run(model, loader, device="cuda", epochs=5, lr=5e-4, kd_T=4.0,
        alpha=0.5, gamma=1.0, msteps=2, **_):
    student = clone_model(model).to(device).train()
    teacher = clone_model(model).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = sgd(student.parameters(), lr, weight_decay=1e-4)
    div = DistillKL(kd_T)

    for epoch in range(epochs):
        # ---- MAX step: push student AWAY from teacher on forget pixels ----
        if epoch < msteps:
            for img, orig, _merged in loader:
                img, orig = img.to(device), orig.to(device)
                forget, _ = pixel_masks(orig)
                if forget.sum() == 0:
                    continue
                with torch.no_grad():
                    t = seg_forward(teacher, img)
                s = seg_forward(student, img)
                loss = -div(s, t, forget)                 # maximise divergence
                opt.zero_grad(); loss.backward()
                # The MAX step is unbounded gradient ascent on a KL divergence;
                # on classes with sparse forget masks per batch (e.g. car, far
                # rarer per-frame than person) the per-batch loss/gradient can
                # have high variance and this diverges to NaN without clipping.
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
                opt.step()
        # ---- MIN step: keep student close to teacher + correct on retain ----
        for img, orig, _merged in loader:
            img, orig = img.to(device), orig.to(device)
            _, retain = pixel_masks(orig)
            with torch.no_grad():
                t = seg_forward(teacher, img)
            s = seg_forward(student, img)
            loss = gamma * masked_ce(s, orig, retain) + alpha * div(s, t, retain)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            opt.step()
    return student
