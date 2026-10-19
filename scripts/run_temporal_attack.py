"""Temporal leakage protocol -- the headline measurement.

For each model, train BOTH:
  * a per-frame silhouette attacker (spatial leakage I(M_t;O_t)), and
  * a temporal trajectory attacker (temporal leakage I(M_{1:T};O_{1:T})),
each N_TRIALS times, and report mean+/-std AUC/IoU. The oracle gives the floor on
both. A method "leaks temporally" iff its temporal AUC exceeds the temporal
floor+2sigma EVEN WHEN its per-frame AUC sits at the per-frame floor -- that gap
is the paper's thesis (per-frame-safe but trajectory-leaky).

Usage: PYTHONPATH=. python scripts/run_temporal_attack.py
Env: N_TRIALS, UNLEARNED (comma list of checkpoints/unlearned_<name>.pth),
     CLIP_LEN, MAX_ITERS, OUT_JSON, TEMPORAL_KIND, SURFACES (default "frame,temporal";
     pass "temporal" alone to skip the frame attacker, e.g. for a CLIP_LEN sweep where
     the frame numbers don't depend on CLIP_LEN and are already known), BS_CLIP
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from seg.attack import (train_attacker, eval_attacker, train_temporal_attacker,
                        eval_temporal_attacker)
from seg.data import VSPWUnlearn, VSPWClip, PERSON
from seg.model import get_seg_model

N_TRIALS = int(os.environ.get("N_TRIALS", "3"))
CLIP_LEN = int(os.environ.get("CLIP_LEN", "3"))
MAX_ITERS = int(os.environ.get("MAX_ITERS", "500"))
TEMPORAL_KIND = os.environ.get("TEMPORAL_KIND", "conv3d")
BS_FRAME = 12
BS_CLIP = int(os.environ.get("BS_CLIP", "6"))
SURFACES = tuple(s for s in os.environ.get("SURFACES", "frame,temporal").split(",") if s)
N_VAL = int(os.environ.get("N_VAL", "1500"))          # cap clips/frames for speed
ARCH = os.environ.get("ARCH", "deeplabv3r50")


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = get_seg_model(device=device, arch=ARCH); m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def build_models():
    models = [
        ("oracle_FLOOR",     f"checkpoints/vspw_oracle_{ARCH}_cls{PERSON}.pth", None),
        ("original",         f"checkpoints/vspw_original_{ARCH}_cls{PERSON}.pth", None),
        ("DELETE_logitmask", f"checkpoints/vspw_original_{ARCH}_cls{PERSON}.pth", "mask"),
    ]
    for name in [s for s in os.environ.get("UNLEARNED", "").split(",") if s]:
        models.append((name, f"checkpoints/unlearned_{name}.pth", None))
    return [(n, p, t) for n, p, t in models if os.path.exists(p) or _warn(n, p)]


def _warn(n, p):
    print(f"[skip] {n}: {p} missing", flush=True); return False


def main():
    device = "cuda"
    # per-frame loaders (frame attacker) + clip loaders (temporal attacker)
    uf_tr = VSPWUnlearn("train", size=512, crop=False)
    uf_va = VSPWUnlearn("val", size=512, crop=False)
    cf_tr = VSPWClip("train", size=512, clip_len=CLIP_LEN, stride=CLIP_LEN,
                     crop=False, person_only=True)
    cf_va = VSPWClip("val", size=512, clip_len=CLIP_LEN, stride=CLIP_LEN,
                     crop=False, person_only=True)
    if N_VAL > 0:
        uf_tr = Subset(uf_tr, list(range(min(N_VAL * 4, len(uf_tr)))))
        uf_va = Subset(uf_va, list(range(min(N_VAL, len(uf_va)))))
        cf_tr = Subset(cf_tr, list(range(min(N_VAL * 4, len(cf_tr)))))
        cf_va = Subset(cf_va, list(range(min(N_VAL, len(cf_va)))))
    ftl = DataLoader(uf_tr, batch_size=BS_FRAME, shuffle=True, num_workers=12, pin_memory=True)
    fvl = DataLoader(uf_va, batch_size=BS_FRAME, shuffle=False, num_workers=8, pin_memory=True)
    ctl = DataLoader(cf_tr, batch_size=BS_CLIP, shuffle=True, num_workers=12, pin_memory=True)
    cvl = DataLoader(cf_va, batch_size=BS_CLIP, shuffle=False, num_workers=8, pin_memory=True)

    rows = {}
    for name, path, transform in build_models():
        model = load(path, device)
        rec = {}
        for surf in SURFACES:
            aucs, ious, cons = [], [], []
            for t in range(N_TRIALS):
                if surf == "frame":
                    D = train_attacker(model, ftl, device, transform=transform,
                                       epochs=10 ** 6, max_iters=MAX_ITERS,
                                       width=128, depth=6, dilation=2)
                    r = eval_attacker(D, model, fvl, device, transform=transform)
                else:
                    D = train_temporal_attacker(model, ctl, device, transform=transform,
                                                epochs=10 ** 6, max_iters=MAX_ITERS,
                                                kind=TEMPORAL_KIND)
                    r = eval_temporal_attacker(D, model, cvl, device, transform=transform)
                    cons.append(r.get("traj_consistency", float("nan")))
                aucs.append(r["attacker_AUC"]); ious.append(r["attacker_IoU"])
            rec[surf] = {
                "AUC_mean": round(float(np.nanmean(aucs)), 1),
                "AUC_std": round(float(np.nanstd(aucs)), 1),
                "IoU_mean": round(float(np.nanmean(ious)), 1),
            }
            if surf == "temporal":
                rec[surf]["traj_consistency"] = round(float(np.nanmean(cons)), 1)
            print(f"  {name:18s} {surf:8s}: AUC {rec[surf]['AUC_mean']:.1f}"
                  f"+/-{rec[surf]['AUC_std']:.1f} IoU {rec[surf]['IoU_mean']:.1f}"
                  + (f" cons {rec[surf].get('traj_consistency')}" if surf == "temporal" else ""),
                  flush=True)
        rows[name] = rec
        del model; torch.cuda.empty_cache()

    out = os.environ.get("OUT_JSON", "results/seg/temporal_attack.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2)

    if "oracle_FLOOR" in rows and "temporal" in SURFACES:
        tf = rows["oracle_FLOOR"]["temporal"]
        have_frame = "frame" in SURFACES
        if have_frame:
            ff = rows["oracle_FLOOR"]["frame"]
            print(f"\nfloors: frame_AUC {ff['AUC_mean']}+/-{ff['AUC_std']}  "
                  f"temporal_AUC {tf['AUC_mean']}+/-{tf['AUC_std']}")
        else:
            print(f"\nfloor: temporal_AUC {tf['AUC_mean']}+/-{tf['AUC_std']} "
                  "(frame surface skipped this run, see the CLIP_LEN=3 baseline json for it)")
        print("a method LEAKS on a surface iff its AUC > floor_mean + 2*floor_std")
        for name, r in rows.items():
            if name == "oracle_FLOOR":
                continue
            tl = "LEAK" if r["temporal"]["AUC_mean"] > tf["AUC_mean"] + 2 * tf["AUC_std"] else "safe"
            if have_frame:
                fl = "LEAK" if r["frame"]["AUC_mean"] > ff["AUC_mean"] + 2 * ff["AUC_std"] else "safe"
                gap = r["temporal"]["AUC_mean"] - r["frame"]["AUC_mean"]
                print(f"  {name:18s} frame:{fl:4s} temporal:{tl:4s} (T-F gap {gap:+.1f})")
            else:
                print(f"  {name:18s} temporal:{tl:4s}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
