"""Train the reference models for car-unlearning on VKITTI2 (second dataset,
cross-dataset generalization check alongside VSPW person/car).

  --mode original : DeepLabV3-R50 per-frame on VKITTI2 with original labels.
  --mode oracle   : same, but Car pixels dominant-neighbor relabeled per-frame.

Usage:
  PYTHONPATH=. python scripts/train_seg_vkitti.py --mode original --epochs 30 --bs 6
  PYTHONPATH=. python scripts/train_seg_vkitti.py --mode oracle   --epochs 30 --bs 6
"""
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from seg.model import get_seg_model
from seg.vkitti_data import VKittiSeg, CARCLS, NUM_CLASSES
from seg.vkitti_engine import seg_metrics, train_seg
from seg.engine import setup_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["original", "oracle"], required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--arch", default="deeplabv3r50")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    setup_seed(args.seed)
    target = "orig" if args.mode == "original" else "merged"
    tr = VKittiSeg("train", target=target, size=args.size)
    va = VKittiSeg("val", target="orig", size=args.size, crop=False)
    tl = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=16,
                    pin_memory=True, drop_last=True)
    vl = DataLoader(va, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    model = get_seg_model(num_classes=NUM_CLASSES, device="cuda", arch=args.arch)
    tag = f"vkitti_{args.mode}_{args.arch}_cls{CARCLS}"
    print(f"[{tag}] {len(tr)} train frames, {len(va)} val frames, "
          f"{args.epochs} epochs, bs{args.bs} res{args.size}, num_classes={NUM_CLASSES}")
    t0 = time.time()
    train_seg(model, tl, args.epochs, args.lr, device="cuda",
              log_prefix=f"[{tag}] ", val_loader=vl, log_every=2)
    hrs = (time.time() - t0) / 3600

    os.makedirs(args.ckpt_dir, exist_ok=True)
    path = os.path.join(args.ckpt_dir, tag + ".pth")
    torch.save({"state_dict": model.state_dict(), "args": vars(args),
                "num_classes": NUM_CLASSES}, path)
    m = seg_metrics(model, vl, "cuda")
    print(f"[{tag}] DONE {hrs:.1f}h -> {path}")
    print(f"[{tag}] val car_IoU {m['car_IoU']:.1f} retain_mIoU {m['retain_mIoU']:.1f} mIoU {m['mIoU']:.1f}")


if __name__ == "__main__":
    main()
