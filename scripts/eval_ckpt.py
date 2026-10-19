"""Clean eval of a saved checkpoint (fresh process). Per-frame seg metrics +
leakage metrics + relearning recovery -- the project1 eval_ckpt, VSPW loaders.

Usage:
  PYTHONPATH=. python scripts/eval_ckpt.py --ckpt checkpoints/unlearned_ADVR-T.pth
  PYTHONPATH=. python scripts/eval_ckpt.py --ref oracle   # eval a reference model
"""
import argparse

import torch
from torch.utils.data import DataLoader, Subset

from seg.audit import leakage_metrics, relearn_audit
from seg.data import VSPWSeg, VSPWUnlearn, PERSON
from seg.engine import seg_metrics
from seg.model import get_seg_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="path to a checkpoint .pth")
    ap.add_argument("--ref", default=None, choices=["original", "oracle"],
                    help="shortcut for checkpoints/vspw_<ref>_deeplabv3r50_cls60.pth")
    ap.add_argument("--arch", default="deeplabv3r50")
    ap.add_argument("--val_cap", type=int, default=4000,
                    help="cap val/unlearn frames for a fast eval (0 = all)")
    ap.add_argument("--relearn_steps", type=int, default=50)
    args = ap.parse_args()
    device = "cuda"

    path = args.ckpt or f"checkpoints/vspw_{args.ref}_deeplabv3r50_cls{PERSON}.pth"
    ck = torch.load(path, map_location=device, weights_only=False)
    m = get_seg_model(device=device, arch=args.arch); m.load_state_dict(ck["state_dict"])

    va = VSPWSeg("val", "orig", size=512, crop=False)
    uv = VSPWUnlearn("val", size=512, crop=False)
    ut = VSPWUnlearn("train", size=512, crop=True)
    if args.val_cap > 0:
        va = Subset(va, list(range(min(args.val_cap, len(va)))))
        uv = Subset(uv, list(range(min(args.val_cap, len(uv)))))
    vl = DataLoader(va, batch_size=6, num_workers=8, pin_memory=True)
    uvl = DataLoader(uv, batch_size=6, num_workers=8, pin_memory=True)
    ul = DataLoader(ut, batch_size=6, shuffle=True, num_workers=8, pin_memory=True,
                    drop_last=True)

    r = seg_metrics(m, vl, device)
    r.update(leakage_metrics(m, uvl, device))
    au = relearn_audit(m, ul, device=device, steps=args.relearn_steps, val_loader=vl)
    name = args.ckpt or args.ref
    print(f"{name}: person {r['person_IoU']:.1f} | retain {r['retain_mIoU']:.1f} "
          f"| blend_fail {r['blend_fail']:.1f} | silhouette {r['silhouette_IoU']:.1f} "
          f"| recover {au.get('recover_pts')}")


if __name__ == "__main__":
    main()
