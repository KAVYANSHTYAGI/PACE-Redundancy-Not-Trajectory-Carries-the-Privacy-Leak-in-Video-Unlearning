"""Utility + per-frame-leakage benchmark table over all checkpoints.

Evaluates the references (original, oracle) and every checkpoints/unlearned_*.pth
on: person_IoU, retain_mIoU, blend_fail, silhouette_IoU, recover_pts. Writes
results/seg/benchmark.json. Pair with run_temporal_attack.py (temporal leakage)
for the complete results section: this file = "did it forget + preserve utility
per-frame", that file = "did the trajectory leak".

Usage: PYTHONPATH=. python scripts/run_benchmark.py
Env: VAL_CAP (default 4000), RELEARN_STEPS (default 50), OUT_JSON
"""
import glob
import json
import os

import torch
from torch.utils.data import DataLoader, Subset

from seg.audit import leakage_metrics, relearn_audit
from seg.data import VSPWSeg, VSPWUnlearn, PERSON
from seg.engine import seg_metrics
from seg.model import get_seg_model

ARCH = os.environ.get("ARCH", "deeplabv3r50")
VAL_CAP = int(os.environ.get("VAL_CAP", "4000"))
RELEARN_STEPS = int(os.environ.get("RELEARN_STEPS", "50"))


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = get_seg_model(device=device, arch=ARCH); m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def main():
    device = "cuda"
    va = VSPWSeg("val", "orig", size=512, crop=False)
    uv = VSPWUnlearn("val", size=512, crop=False)
    ut = VSPWUnlearn("train", size=512, crop=True)
    if VAL_CAP > 0:
        va = Subset(va, list(range(min(VAL_CAP, len(va)))))
        uv = Subset(uv, list(range(min(VAL_CAP, len(uv)))))
    vl = DataLoader(va, batch_size=6, num_workers=8, pin_memory=True)
    uvl = DataLoader(uv, batch_size=6, num_workers=8, pin_memory=True)
    ul = DataLoader(ut, batch_size=6, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    targets = []
    for ref in ("original", "oracle"):
        p = f"checkpoints/vspw_{ref}_{ARCH}_cls{PERSON}.pth"
        if os.path.exists(p):
            targets.append((ref, p))
    for p in sorted(glob.glob("checkpoints/unlearned_*.pth")):
        targets.append((os.path.basename(p)[len("unlearned_"):-4], p))

    rows = {}
    for name, path in targets:
        m = load(path, device)
        r = seg_metrics(m, vl, device)
        r.update(leakage_metrics(m, uvl, device))
        au = relearn_audit(m, ul, device=device, steps=RELEARN_STEPS, val_loader=vl)
        r["recover_pts"] = au.get("recover_pts")
        rows[name] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}
        print(f"  {name:22s} person {r['person_IoU']:5.1f} | retain {r['retain_mIoU']:5.1f} "
              f"| blend {r['blend_fail']:5.1f} | silh {r['silhouette_IoU']:5.1f} "
              f"| recover {r['recover_pts']}", flush=True)
        del m; torch.cuda.empty_cache()

    out = os.environ.get("OUT_JSON", "results/seg/benchmark.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
