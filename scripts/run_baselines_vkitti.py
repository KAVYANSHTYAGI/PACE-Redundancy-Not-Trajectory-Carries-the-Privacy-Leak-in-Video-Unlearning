"""Train the published unlearning baselines on VKITTI2 (forget class: Car),
mirroring scripts/run_baselines.py exactly. The baseline algorithms
(seg/baselines/*) read PERSON/NUM_CLASSES/IGNORE from seg.data at import time,
which are env-var driven (FORGET_CLS, NUM_CLASSES) -- so as long as those are
set to VKITTI2's Car index (1) and class count (14) before this script's first
import, the same baseline code runs unmodified against the new dataset. Only
the data source (VKittiUnlearn instead of VSPWUnlearn) and checkpoint paths
differ.

Usage:
  VK_FORGET_CLS=1 PYTHONPATH=. python scripts/run_baselines_vkitti.py
  (must also export FORGET_CLS=1 NUM_CLASSES=14 -- see run_vkitti_baselines.sh)
"""
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from seg.baselines import BASELINES, run_baseline
from seg.data import PERSON  # env-driven; must equal VKITTI2's CARCLS (1)
from seg.engine import setup_seed
from seg.model import get_seg_model
from seg.vkitti_data import VKittiUnlearn, CARCLS, NUM_CLASSES

HP = {
    "FT":        dict(epochs=4, lr=0.01),
    "NegGrad":   dict(epochs=2, lr=1e-4),
    "NegGrad+":  dict(epochs=2, lr=0.01, beta=0.999),
    "RL":        dict(epochs=4, lr=0.013),
    "SalUn":     dict(epochs=4, lr=0.013, threshold=0.5),
    "SSD":       dict(selection_weighting=10.0, dampening_constant=1.0),
    "SCRUB":     dict(epochs=4, lr=5e-4, msteps=1),
    "DELETE":    dict(epochs=4, lr=0.01),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--arch", default="deeplabv3r50")
    args = ap.parse_args()
    device = "cuda"
    setup_seed(args.seed)

    assert PERSON == CARCLS, (
        f"seg.data.PERSON ({PERSON}) != VKITTI2 CARCLS ({CARCLS}) -- "
        f"FORGET_CLS env var must be set to {CARCLS} before this script runs")

    orig_ck = f"checkpoints/vkitti_original_{args.arch}_cls{CARCLS}.pth"
    ck = torch.load(orig_ck, map_location=device, weights_only=False)
    base = get_seg_model(num_classes=NUM_CLASSES, device=device, arch=args.arch)
    base.load_state_dict(ck["state_dict"])

    un = VKittiUnlearn("train", size=512, crop=True)
    ul = DataLoader(un, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True)

    names = args.only or list(BASELINES)
    os.makedirs("checkpoints", exist_ok=True)
    for name in names:
        out = f"checkpoints/unlearned_bl_{name}_s{args.seed}_vkitti.pth"
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
