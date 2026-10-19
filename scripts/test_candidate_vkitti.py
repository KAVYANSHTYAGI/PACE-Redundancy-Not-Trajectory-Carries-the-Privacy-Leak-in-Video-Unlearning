"""Train FAE/TAE/PACE on VKITTI2 (forget class: Car), mirroring
scripts/test_candidate.py. seg/candidates.py reads PERSON/NUM_CLASSES/IGNORE
from seg.data at import time (env-driven), so the same generator/adversary
code runs unmodified against VKITTI2 as long as FORGET_CLS=1 NUM_CLASSES=14
are set before import. Only the data source and checkpoint naming differ; the
VSPW-specific eval tail is dropped (use scripts/utility_check-style scripts
and the pooled-attacker protocol separately, as done for VSPW/car).

Usage:
  FORGET_CLS=1 NUM_CLASSES=14 PYTHONPATH=. python scripts/test_candidate_vkitti.py \
      --name ADVR-T --epochs 30 --clip_len 3 --lam_tc 0.5 --save_as TAE_vkitti
"""
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from seg.candidates import run_candidate, CANDIDATES
from seg.data import PERSON  # env-driven; must equal VKITTI2's CARCLS (1)
from seg.engine import setup_seed
from seg.model import get_seg_model
from seg.vkitti_data import VKittiUnlearn, VKittiClip, VKittiSeg, CARCLS, NUM_CLASSES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, choices=list(CANDIDATES))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam_r", type=float, default=1.0)
    p.add_argument("--lam_adv", type=float, default=1.0)
    p.add_argument("--lam_tc", type=float, default=0.0)
    p.add_argument("--lam_pool", type=float, default=1.0)
    p.add_argument("--lr_d", type=float, default=1e-3)
    p.add_argument("--out_size", type=int, default=128)
    p.add_argument("--temporal_kind", default="conv3d", choices=["conv3d", "convgru", "mamba"])
    p.add_argument("--clip_len", type=int, default=3)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no_replay", action="store_true")
    p.add_argument("--save_as", default="")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--arch", default="deeplabv3r50")
    args = p.parse_args()
    device = "cuda"
    setup_seed(args.seed)

    assert PERSON == CARCLS, (
        f"seg.data.PERSON ({PERSON}) != VKITTI2 CARCLS ({CARCLS}) -- "
        f"FORGET_CLS env var must be set to {CARCLS} before this script runs")

    orig_ck = f"checkpoints/vkitti_original_{args.arch}_cls{CARCLS}.pth"
    ck = torch.load(orig_ck, map_location=device, weights_only=False)
    m = get_seg_model(num_classes=NUM_CLASSES, device=device, arch=args.arch)
    m.load_state_dict(ck["state_dict"])

    is_temporal = args.name in ("ADVR-T", "ADVR-TX", "ADVR-P")
    if is_temporal:
        ds = VKittiClip("train", size=args.size, clip_len=args.clip_len,
                        stride=args.clip_len, crop=True, car_only=True)
        ul = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True)
    else:
        ds = VKittiUnlearn("train", size=args.size, crop=True)
        ul = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True)

    replay_loader = None
    if not args.no_replay:
        rep = VKittiSeg("train", "orig", size=args.size, crop=True)
        replay_loader = DataLoader(rep, batch_size=args.bs, shuffle=True,
                                   num_workers=args.workers, pin_memory=True, drop_last=True)

    kw = dict(epochs=args.epochs, lr=args.lr, lam_r=args.lam_r, lam_adv=args.lam_adv,
              lr_d=args.lr_d, out_size=args.out_size, replay_loader=replay_loader)
    if is_temporal:
        kw.update(lam_tc=args.lam_tc, temporal_kind=args.temporal_kind)
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


if __name__ == "__main__":
    main()
