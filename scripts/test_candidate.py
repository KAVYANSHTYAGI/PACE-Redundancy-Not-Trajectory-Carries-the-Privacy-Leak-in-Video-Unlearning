"""Train an ADVR / ADVR-T candidate on VSPW, save, and (optionally) eval.

ADVR   consumes per-frame VSPWUnlearn batches (img, orig, merged) + a retain-replay
       loader of full-distribution frames.
ADVR-T consumes VSPWClip batches (imgs[N,T,...], origs, mergeds) + the same
       per-frame retain-replay loader.

Usage:
  PYTHONPATH=. python scripts/test_candidate.py --name ADVR   --epochs 4 --save_as ADVR
  PYTHONPATH=. python scripts/test_candidate.py --name ADVR-T --epochs 4 --clip_len 3 \
      --lam_adv 1.0 --lam_tc 0.5 --save_as ADVR-T
"""
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from seg.candidates import run_candidate, CANDIDATES
from seg.data import VSPWUnlearn, VSPWClip, VSPWSeg, PERSON
from seg.engine import setup_seed
from seg.model import get_seg_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, choices=list(CANDIDATES))
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam_r", type=float, default=1.0)
    p.add_argument("--lam_adv", type=float, default=1.0)
    p.add_argument("--lam_tc", type=float, default=0.0, help="ADVR-T/-TX temporal-consistency weight")
    p.add_argument("--lam_ord", type=float, default=1.0,
                   help="ADVR-TX only: temporal-exchangeability adversary weight. "
                        "Set 0 to ablate OrderDiscriminator out of ADVR-TX (see "
                        "EXPERIMENTS.md sec 8 item 2a).")
    p.add_argument("--lam_pool", type=float, default=1.0,
                   help="ADVR-P only: pooling-discriminator (PoolDiscriminator) "
                        "fooling weight -- attacks Delta_redundancy directly "
                        "(see EXPERIMENTS.md sec 9/10 item 3).")
    p.add_argument("--lr_d", type=float, default=1e-3)
    p.add_argument("--out_size", type=int, default=128)
    p.add_argument("--temporal_kind", default="conv3d", choices=["conv3d", "convgru", "mamba"])
    p.add_argument("--clip_len", type=int, default=3)
    p.add_argument("--bs", type=int, default=6, help="frames (ADVR) or clips (ADVR-T)")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--no_replay", action="store_true")
    p.add_argument("--save_as", default="")
    p.add_argument("--no_eval", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--arch", default="deeplabv3r50")
    args = p.parse_args()
    device = "cuda"
    setup_seed(args.seed)

    orig_ck = f"checkpoints/vspw_original_{args.arch}_cls{PERSON}.pth"
    ck = torch.load(orig_ck, map_location=device, weights_only=False)
    m = get_seg_model(device=device, arch=args.arch); m.load_state_dict(ck["state_dict"])

    is_temporal = args.name in ("ADVR-T", "ADVR-TX", "ADVR-P")
    if is_temporal:
        ds = VSPWClip("train", size=args.size, clip_len=args.clip_len,
                      stride=args.clip_len, crop=True, person_only=True)
        # clips already stack T frames; keep batch small so N*T fits memory
        ul = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True)
    else:
        ds = VSPWUnlearn("train", size=args.size, crop=True)
        ul = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True)

    replay_loader = None
    if not args.no_replay:
        rep = VSPWSeg("train", "orig", size=args.size, crop=True)
        replay_loader = DataLoader(rep, batch_size=args.bs, shuffle=True,
                                   num_workers=args.workers, pin_memory=True, drop_last=True)

    kw = dict(epochs=args.epochs, lr=args.lr, lam_r=args.lam_r, lam_adv=args.lam_adv,
              lr_d=args.lr_d, out_size=args.out_size, replay_loader=replay_loader)
    if is_temporal:
        kw.update(lam_tc=args.lam_tc, temporal_kind=args.temporal_kind)
    if args.name == "ADVR-TX":
        kw.update(lam_ord=args.lam_ord)
    if args.name == "ADVR-P":
        kw.update(lam_pool=args.lam_pool)

    print(f"[{args.name}] {len(ds)} train units, epochs={args.epochs}, "
          f"{'clip_len='+str(args.clip_len) if is_temporal else 'per-frame'}", flush=True)
    t0 = time.time()
    um = run_candidate(args.name, m, ul, device=device, **kw)
    dt = time.time() - t0

    if args.save_as:
        os.makedirs("checkpoints", exist_ok=True)
        out = f"checkpoints/unlearned_{args.save_as}.pth"
        torch.save({"state_dict": um.state_dict(), "config": {**kw, "name": args.name}}, out)
        print(f"  saved {out}")
    print(f"[{args.name}] trained in {dt/60:.1f}min", flush=True)

    if args.no_eval:
        return
    from seg.audit import leakage_metrics
    from seg.engine import seg_metrics
    from torch.utils.data import Subset
    va = Subset(VSPWSeg("val", "orig", size=512, crop=False), list(range(4000)))
    uv = Subset(VSPWUnlearn("val", size=512, crop=False), list(range(4000)))
    vl = DataLoader(va, batch_size=6, num_workers=8, pin_memory=True)
    uvl = DataLoader(uv, batch_size=6, num_workers=8, pin_memory=True)
    row = seg_metrics(um, vl, device); row.update(leakage_metrics(um, uvl, device))
    print(f"[{args.name}] person {row['person_IoU']:.1f} | retain {row['retain_mIoU']:.1f} "
          f"| blend_fail {row['blend_fail']:.1f} | silhouette {row['silhouette_IoU']:.1f}")


if __name__ == "__main__":
    main()
