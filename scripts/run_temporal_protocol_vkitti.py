"""Corrected temporal-leakage protocol (supersedes run_temporal_attack.py).

Three defects in the original protocol are fixed here; each of them, on its own,
is enough to invalidate a leakage-vs-clip-length claim.

1. MATCHED EVALUATION CONTENT. VSPWClip tiles videos with stride=clip_len, so
   the clip count collapses with T and a fixed cap silently changed which data
   each T was scored on (val: 36 distinct videos at T=2 vs 238 at T=16). We now
   use VSPWClipAnchored: one anchor set, drawn from the split alone, windowed at
   every T. Diagnostic that the old numbers were confounded: the oracle floor,
   which cannot depend on T, drifted 17.7 -> 15.2 -> 5.8 -> 0.0 -> 0.7.

2. MATCHED ATTACKER BUDGET. Physical batch has to shrink as T grows, and with a
   fixed iteration count that handed the attacker 8x less data (and batch-size-1
   gradients) at T=16. Leakage then falls with T for every model for free. We
   now fix the number of optimizer steps AND the effective batch in clips at
   every T, using gradient accumulation, and log the attacker's final training
   loss so a converged-but-empty attacker is distinguishable from one that never
   trained.

3. THRESHOLD-FREE SCORING. IoU@0.5 collapses to 0.0 for an informative but
   under-confident attacker (ADVR-T at T=16 scored IoU 0.0 at AUC 61.2, i.e.
   above chance). Primary metric is now AP, with IoU_best as the calibration-
   free reconstruction quality and pos_rate to expose all-negative collapse.

It also adds the control the central claim actually needs:

   TIME-SHUFFLED ARM. A temporal attacker can beat a per-frame attacker for a
   boring reason -- averaging T correlated per-frame estimates reduces variance
   -- or for the interesting one, information carried by the trajectory itself.
   Shuffling the clip's time order (same permutation on fields and masks) kills
   the second and leaves the first untouched. So
        synergy = AP(ordered) - AP(shuffled)
   is the part of temporal leakage that is genuinely trajectory-borne. If
   synergy ~ 0, "temporal silhouette leakage" reduces to denoising and the
   thesis needs restating; if it is clearly positive on per-frame-defended
   models and ~0 on ADVR-T, that is the paper.

Usage: PYTHONPATH=. python scripts/run_temporal_protocol.py
Env: CLIP_LEN, N_TRIALS, STEPS, EFF_CLIPS, N_EVAL, N_TRAIN, MODELS, OUT_JSON,
     TEMPORAL_KIND, WITH_FRAME (1 to also run the per-frame surface)
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from seg.attack import (train_attacker, eval_attacker, train_temporal_attacker,
                        eval_temporal_attacker)
from seg.vkitti_data import VKittiClipAnchored, VKittiUnlearn, CARCLS, NUM_CLASSES
from seg.data import PERSON  # env-driven; must equal CARCLS, see assert in main()
from seg.model import get_seg_model

CLIP_LEN = int(os.environ.get("CLIP_LEN", "3"))
N_TRIALS = int(os.environ.get("N_TRIALS", "2"))
STEPS = int(os.environ.get("STEPS", "750"))       # optimizer steps, same at every T
EFF_CLIPS = int(os.environ.get("EFF_CLIPS", "4"))  # effective batch in clips, same at every T
N_EVAL = int(os.environ.get("N_EVAL", "1200"))     # anchors held for eval
N_TRAIN = int(os.environ.get("N_TRAIN", "6000"))   # anchors for attacker training
TEMPORAL_KIND = os.environ.get("TEMPORAL_KIND", "conv3d")
WITH_FRAME = os.environ.get("WITH_FRAME", "0") == "1"
ARCH = os.environ.get("ARCH", "deeplabv3r50")
MIN_VIDEO_LEN = int(os.environ.get("MIN_VIDEO_LEN", "16"))
# frame_stride>1 dilates the clip in time: trajectory information scales with
# ELAPSED TIME, not frame count (16 consecutive VSPW frames span only ~1.1s).
FRAME_STRIDE = int(os.environ.get("FRAME_STRIDE", "1"))

# Physical batch that fits 16GB at this T; accumulation restores EFF_CLIPS.
_PHYS = {2: 4, 3: 4, 5: 2, 8: 2, 16: 1}
PHYS_BS = int(os.environ.get("PHYS_BS", str(_PHYS.get(CLIP_LEN, 1))))
ACCUM = max(1, EFF_CLIPS // PHYS_BS)

DEFAULT_MODELS = "oracle_FLOOR,original,ADVR,ADVR-T,bl_SalUn_s1"


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = get_seg_model(num_classes=NUM_CLASSES, device=device, arch=ARCH)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def build_models():
    named = {
        "oracle_FLOOR": (f"checkpoints/vkitti_oracle_{ARCH}_cls{CARCLS}.pth", None),
        "original": (f"checkpoints/vkitti_original_{ARCH}_cls{CARCLS}.pth", None),
    }
    out = []
    for name in [s for s in os.environ.get("MODELS", DEFAULT_MODELS).split(",") if s]:
        path, tf = named.get(name, (f"checkpoints/unlearned_{name}.pth", None))
        if os.path.exists(path):
            out.append((name, path, tf))
        else:
            print(f"[skip] {name}: {path} missing", flush=True)
    return out


def agg(rows, key):
    """Mean and SAMPLE std (ddof=1) of `key` across trial dicts.

    BUG FIXED 2026-07-30: this used np.std's default ddof=0 (population std),
    which is the wrong estimator here -- these are N_TRIALS independent
    training/eval runs, a sample, and the population they're drawn from has
    unknown variance, so the unbiased ddof=1 estimator is the correct one. At
    N=2 (used throughout every sweep so far) ddof=0 understates the true
    sample std by a factor of sqrt(2) (1.414x), which directly tightens the
    "leaks iff AP > floor_mean + 2*floor_std" threshold below in main() --
    every LEAK/safe verdict computed so far used a threshold ~41% too low.
    ddof=1 is undefined for a single sample; fall back to 0.0 there (matches
    the old behavior when N=1, which is the only case ddof=0 and ddof=1 must
    differ on by convention rather than by choice).
    """
    vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    if not vals:
        return float("nan"), float("nan")
    ddof = 1 if len(vals) > 1 else 0
    return float(np.mean(vals)), float(np.std(vals, ddof=ddof))


def main():
    device = "cuda"
    assert PERSON == CARCLS, (
        f"seg.data.PERSON ({PERSON}) != VKITTI2 CARCLS ({CARCLS}) -- "
        f"FORGET_CLS env var must be set to {CARCLS} before this script runs")
    ctr = VKittiClipAnchored("train", size=512, clip_len=CLIP_LEN, crop=False,
                             max_anchors=N_TRAIN, seed=0, min_video_len=MIN_VIDEO_LEN,
                             frame_stride=FRAME_STRIDE)
    cva = VKittiClipAnchored("val", size=512, clip_len=CLIP_LEN, crop=False,
                             max_anchors=N_EVAL, seed=0, min_video_len=MIN_VIDEO_LEN,
                             frame_stride=FRAME_STRIDE)
    ctl = DataLoader(ctr, batch_size=PHYS_BS, shuffle=True, num_workers=10,
                     pin_memory=True, drop_last=True)
    cvl = DataLoader(cva, batch_size=PHYS_BS, shuffle=False, num_workers=8, pin_memory=True)
    span_s = ((CLIP_LEN - 1) * FRAME_STRIDE + 1) / 15.0
    print(f"[protocol] T={CLIP_LEN} stride={FRAME_STRIDE} (~{span_s:.1f}s span) "
          f"attacker={TEMPORAL_KIND} phys_bs={PHYS_BS} accum={ACCUM} "
          f"(eff {PHYS_BS*ACCUM} clips) steps={STEPS} trials={N_TRIALS} | "
          f"train {len(ctr)} clips / eval {len(cva)} clips", flush=True)

    ftl = fvl = None
    if WITH_FRAME:
        from torch.utils.data import Subset
        uf_tr = VKittiUnlearn("train", size=512, crop=False)
        uf_va = VKittiUnlearn("val", size=512, crop=False)
        uf_tr = Subset(uf_tr, list(range(min(6000, len(uf_tr)))))
        uf_va = Subset(uf_va, list(range(min(1500, len(uf_va)))))
        ftl = DataLoader(uf_tr, batch_size=12, shuffle=True, num_workers=10, pin_memory=True)
        fvl = DataLoader(uf_va, batch_size=12, shuffle=False, num_workers=8, pin_memory=True)

    rows = {}
    for name, path, transform in build_models():
        model = load(path, device)
        rec = {}

        for arm, shuf in (("ordered", False), ("shuffled", True)):
            trials = []
            for _t in range(N_TRIALS):
                D = train_temporal_attacker(
                    model, ctl, device, transform=transform, epochs=10 ** 6,
                    max_iters=STEPS, kind=TEMPORAL_KIND, accum_steps=ACCUM,
                    shuffle_time=shuf)
                r = eval_temporal_attacker(D, model, cvl, device, transform=transform,
                                           shuffle_time=shuf)
                tail = getattr(D, "train_losses", [])[-50:]
                r["final_train_loss"] = float(np.mean(tail)) if tail else float("nan")
                trials.append(r)
                del D; torch.cuda.empty_cache()
            rec[arm] = {}
            for k in ("attacker_AP", "attacker_AUC", "attacker_IoU",
                      "attacker_IoU_best", "attacker_pos_rate",
                      "traj_consistency", "final_train_loss"):
                mu, sd = agg(trials, k)
                rec[arm][k] = round(mu, 2)
                if k in ("attacker_AP", "attacker_AUC"):
                    rec[arm][k + "_std"] = round(sd, 2)
            a = rec[arm]
            print(f"  {name:16s} {arm:9s}: AP {a['attacker_AP']:5.2f}+/-{a['attacker_AP_std']:.2f}"
                  f"  AUC {a['attacker_AUC']:5.2f}  IoU* {a['attacker_IoU_best']:5.2f}"
                  f"  IoU@.5 {a['attacker_IoU']:5.2f}  posrate {a['attacker_pos_rate']:5.2f}"
                  f"  cons {a['traj_consistency']}  loss {a['final_train_loss']:.4f}", flush=True)

        rec["synergy_AP"] = round(rec["ordered"]["attacker_AP"]
                                  - rec["shuffled"]["attacker_AP"], 2)
        print(f"  {name:16s} {'SYNERGY':9s}: dAP {rec['synergy_AP']:+.2f} "
              "(ordered - time-shuffled; >0 = genuinely trajectory-borne)", flush=True)

        if WITH_FRAME:
            trials = []
            for _t in range(N_TRIALS):
                D = train_attacker(model, ftl, device, transform=transform,
                                   epochs=10 ** 6, max_iters=STEPS,
                                   width=128, depth=6, dilation=2)
                trials.append(eval_attacker(D, model, fvl, device, transform=transform))
                del D; torch.cuda.empty_cache()
            rec["frame"] = {}
            for k in ("attacker_AP", "attacker_AUC", "attacker_IoU",
                      "attacker_IoU_best", "attacker_pos_rate"):
                mu, sd = agg(trials, k)
                rec["frame"][k] = round(mu, 2)
            f = rec["frame"]
            print(f"  {name:16s} {'frame':9s}: AP {f['attacker_AP']:5.2f}"
                  f"  AUC {f['attacker_AUC']:5.2f}  IoU* {f['attacker_IoU_best']:5.2f}", flush=True)

        rows[name] = rec
        del model; torch.cuda.empty_cache()

    rows["_config"] = dict(clip_len=CLIP_LEN, frame_stride=FRAME_STRIDE,
                           span_seconds=round(span_s, 2),
                           steps=STEPS, eff_clips=PHYS_BS * ACCUM,
                           phys_bs=PHYS_BS, accum=ACCUM, n_trials=N_TRIALS,
                           n_eval_clips=len(cva), n_train_clips=len(ctr),
                           temporal_kind=TEMPORAL_KIND)
    out = os.environ.get("OUT_JSON", f"results/seg/protocol_T{CLIP_LEN}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2)

    if "oracle_FLOOR" in rows:
        fl = rows["oracle_FLOOR"]["ordered"]
        thr = fl["attacker_AP"] + 2 * fl["attacker_AP_std"]
        print(f"\nfloor(ordered) AP {fl['attacker_AP']:.2f}+/-{fl['attacker_AP_std']:.2f}"
              f"  -> leaks iff AP > {thr:.2f}")
        for name, r in rows.items():
            if name in ("oracle_FLOOR", "_config"):
                continue
            v = "LEAK" if r["ordered"]["attacker_AP"] > thr else "safe"
            print(f"  {name:16s} {v:4s}  AP {r['ordered']['attacker_AP']:5.2f}"
                  f"  synergy {r['synergy_AP']:+.2f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
