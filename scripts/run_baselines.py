"""Train the published unlearning baselines on VSPW and save each as
checkpoints/unlearned_bl_<NAME>.pth. Faithful ports (seg/baselines/), one seed.

Usage:
  PYTHONPATH=. python scripts/run_baselines.py                 # all baselines
  PYTHONPATH=. python scripts/run_baselines.py --only SSD SalUn
Env: FORGET_CLS/NUM_CLASSES inherited from seg.data defaults (person, 124).
"""
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from seg.baselines import BASELINES, run_baseline
from seg.data import VSPWUnlearn, PERSON
from seg.engine import setup_seed
from seg.model import get_seg_model

# per-baseline hyper-params (published defaults, matched to project1). VSPW has
# ~13x more person frames/epoch than ADE20K, so epoch counts are kept low.
HP = {
    "FT":        dict(epochs=2, lr=0.01),
    "NegGrad":   dict(epochs=1, lr=1e-4),
    "NegGrad+":  dict(epochs=1, lr=0.01, beta=0.999),
    "RL":        dict(epochs=2, lr=0.013),
    "SalUn":     dict(epochs=2, lr=0.013, threshold=0.5),
    "SSD":       dict(selection_weighting=10.0, dampening_constant=1.0),
    "SCRUB":     dict(epochs=2, lr=5e-4, msteps=1),
    "DELETE":    dict(epochs=2, lr=0.01),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="subset of baseline names")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--arch", default="deeplabv3r50")
    args = ap.parse_args()
    device = "cuda"
    setup_seed(args.seed)

    orig_ck = f"checkpoints/vspw_original_{args.arch}_cls{PERSON}.pth"
    ck = torch.load(orig_ck, map_location=device, weights_only=False)
    base = get_seg_model(device=device, arch=args.arch); base.load_state_dict(ck["state_dict"])

    un = VSPWUnlearn("train", size=512, crop=True)
    ul = DataLoader(un, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True)

    names = args.only or list(BASELINES)
    os.makedirs("checkpoints", exist_ok=True)
    # NOTE: checkpoint name includes cls{PERSON} (the forget class) -- without
    # this, running a different FORGET_CLS (e.g. car, to test generalization
    # beyond person) would silently overwrite the existing person-class
    # baseline checkpoints, since the old naming had no class in it at all.
    for name in names:
        out = f"checkpoints/unlearned_bl_{name}_s{args.seed}_cls{PERSON}.pth"
        if os.path.exists(out):
            print(f"[skip] {name} ({out} exists)"); continue
        hp = HP.get(name, {})
        print(f"[{name}] training with {hp}", flush=True)
        t0 = time.time()
        um = run_baseline(name, base, ul, device=device, **hp)
        torch.save({"state_dict": um.state_dict(), "baseline": name}, out)
        print(f"[{name}] DONE {(time.time()-t0)/60:.1f}min -> {out}", flush=True)


if __name__ == "__main__":
    main()
