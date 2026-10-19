"""Fast utility check (car_IoU / retain_mIoU / mIoU) across all car-class
checkpoints, no relearn audit -- just seg_metrics on a capped val set. Needed
for the car-class results table (paper); mirrors what train_seg.py already
logged for oracle_car/original_car but baselines/ADVR-T/ADVR-P never had this
run against them.

Usage: PYTHONPATH=. FORGET_CLS=48 python scripts/utility_check_car.py
"""
import json
import os

import torch
from torch.utils.data import DataLoader, Subset

from seg.data import VSPWSeg, PERSON
from seg.engine import seg_metrics
from seg.model import get_seg_model

ARCH = "deeplabv3r50"
MODELS = {
    "oracle_FLOOR": f"checkpoints/vspw_oracle_{ARCH}_cls{PERSON}.pth",
    "original": f"checkpoints/vspw_original_{ARCH}_cls{PERSON}.pth",
    "bl_FT_s1_cls48": "checkpoints/unlearned_bl_FT_s1_cls48.pth",
    "bl_NegGrad_s1_cls48": "checkpoints/unlearned_bl_NegGrad_s1_cls48.pth",
    "bl_NegGrad+_s1_cls48": "checkpoints/unlearned_bl_NegGrad+_s1_cls48.pth",
    "bl_RL_s1_cls48": "checkpoints/unlearned_bl_RL_s1_cls48.pth",
    "bl_SalUn_s1_cls48": "checkpoints/unlearned_bl_SalUn_s1_cls48.pth",
    "bl_SSD_s1_cls48": "checkpoints/unlearned_bl_SSD_s1_cls48.pth",
    "bl_SCRUB_s1_cls48": "checkpoints/unlearned_bl_SCRUB_s1_cls48.pth",
    "bl_DELETE_s1_cls48": "checkpoints/unlearned_bl_DELETE_s1_cls48.pth",
    "ADVR-T_car": "checkpoints/unlearned_ADVR-T_car.pth",
    "ADVR-P_car": "checkpoints/unlearned_ADVR-P_car.pth",
}


def main():
    device = "cuda"
    va = VSPWSeg("val", "orig", size=512, crop=False)
    va = Subset(va, list(range(min(4000, len(va)))))
    vl = DataLoader(va, batch_size=6, num_workers=8, pin_memory=True)

    out = {}
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} missing", flush=True)
            continue
        ck = torch.load(path, map_location=device, weights_only=False)
        m = get_seg_model(device=device, arch=ARCH)
        m.load_state_dict(ck["state_dict"])
        r = seg_metrics(m, vl, device)
        out[name] = r
        print(f"{name:20s} car_IoU {r['person_IoU']:5.1f}  retain_mIoU {r['retain_mIoU']:5.1f}"
              f"  mIoU {r['mIoU']:5.1f}", flush=True)
        del m
        torch.cuda.empty_cache()

    os.makedirs("results/seg", exist_ok=True)
    json.dump(out, open("results/seg/utility_car.json", "w"), indent=2)
    print("wrote results/seg/utility_car.json")


if __name__ == "__main__":
    main()
