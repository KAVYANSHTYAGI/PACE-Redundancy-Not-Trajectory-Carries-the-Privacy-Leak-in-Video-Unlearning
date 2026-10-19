"""DELETE (Decoupled Distillation to Erase): faithful reproduction of Zhou et al.,
CVPR 2025 (arXiv 2503.23751).

No official code was locatable, so this reproduces the paper's equations directly:

  Mask_u(v) = (1 - e_u) (.) v           # zero the forget-class channel  (Eq. 5)
  target    = Normalize(Mask_u(Softmax(z_teacher)))   # renormalise over remaining
  L         = KL( target || Softmax(z_student) )      # mask-distillation (Eq. 5)

The decoupled loss (Eq. 2) = a forgetting term driving the forget-class prob -> 0
(implicit here, since the target's forget-class prob is exactly 0) + a retention term
that preserves the *relative* probabilities of the remaining classes ("dark
knowledge"), which the renormalised teacher target provides. Teacher = frozen
original; the student is trained by distillation, unlike a naive post-hoc logit mask.

Segmentation adaptation: the distillation is applied per-pixel over all non-ignore
pixels of each image.
"""
import torch
import torch.nn.functional as F

from seg.data import IGNORE, PERSON
from seg.model import seg_forward
from ._common import clone_model, sgd


def run(model, loader, device="cuda", epochs=5, lr=0.01, T=1.0, **_):
    student = clone_model(model).to(device).train()
    teacher = clone_model(model).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = sgd(student.parameters(), lr)

    for _e in range(epochs):
        for img, orig, _merged in loader:
            img, orig = img.to(device), orig.to(device)
            with torch.no_grad():
                t = F.softmax(seg_forward(teacher, img) / T, dim=1)   # [N,C,H,W]
                t[:, PERSON] = 0.0                                     # Mask_u
                t = t / t.sum(1, keepdim=True).clamp(min=1e-8)         # Normalize
            s_logp = F.log_softmax(seg_forward(student, img) / T, dim=1)
            # KL(target || student) per pixel, averaged over valid pixels
            kl = F.kl_div(s_logp, t, reduction="none").sum(1) * (T ** 2)   # [N,H,W]
            valid = (orig != IGNORE)
            loss = (kl * valid).sum() / valid.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
    return student
