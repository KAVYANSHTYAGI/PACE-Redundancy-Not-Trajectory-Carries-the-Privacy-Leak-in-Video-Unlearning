"""Faithful segmentation-unlearning baselines, one module per method.

Each `run(model, loader, device, **hp)` returns an unlearned copy of `model`.
`loader` yields (image, orig_label, merged_label). Sources are cited in each module.
"""
from . import (delete, finetune, neggrad, neggrad_plus, random_label, salun, scrub,
               ssd)

BASELINES = {
    "FT": finetune.run,          # SalUn FT.py
    "NegGrad": neggrad.run,      # SalUn GA.py (gradient ascent)
    "NegGrad+": neggrad_plus.run,  # SCRUB repo train_negrad
    "RL": random_label.run,      # SalUn RL.py
    "SalUn": salun.run,          # SalUn generate_mask + RL (ICLR'24)
    "SSD": ssd.run,              # SSD ParameterPerturber (AAAI'24)
    "SCRUB": scrub.run,          # SCRUB (NeurIPS'23)
    "DELETE": delete.run,        # DELETE mask-distillation (CVPR'25)
}


def run_baseline(name, model, loader, device="cuda", **kw):
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; have {list(BASELINES)}")
    return BASELINES[name](model, loader, device=device, **kw)
