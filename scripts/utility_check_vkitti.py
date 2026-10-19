"""Fast utility check (car_IoU / retain_mIoU / mIoU) across all VKITTI2
checkpoints, mirroring scripts/utility_check_car.py / utility_check_person.py.

Usage: FORGET_CLS=1 NUM_CLASSES=14 PYTHONPATH=. python scripts/utility_check_vkitti.py
"""
import json
import os

import torch
from torch.utils.data import DataLoader, Subset

from seg.data import PERSON
from seg.model import get_seg_model
from seg.vkitti_data import VKittiSeg, CARCLS, NUM_CLASSES
from seg.vkitti_engine import seg_metrics

ARCH = "deeplabv3r50"
MODELS = {
    "oracle_FLOOR": f"checkpoints/vkitti_oracle_{ARCH}_cls{CARCLS}.pth",
    "original": f"checkpoints/vkitti_original_{ARCH}_cls{CARCLS}.pth",
    "bl_FT_s1_vkitti": "checkpoints/unlearned_bl_FT_s1_vkitti.pth",
    "bl_NegGrad_s1_vkitti": "checkpoints/unlearned_bl_NegGrad_s1_vkitti.pth",
    "bl_NegGrad+_s1_vkitti": "checkpoints/unlearned_bl_NegGrad+_s1_vkitti.pth",
    "bl_RL_s1_vkitti": "checkpoints/unlearned_bl_RL_s1_vkitti.pth",
    "bl_SalUn_s1_vkitti": "checkpoints/unlearned_bl_SalUn_s1_vkitti.pth",
    "bl_SSD_s1_vkitti": "checkpoints/unlearned_bl_SSD_s1_vkitti.pth",
    "bl_SCRUB_s1_vkitti": "checkpoints/unlearned_bl_SCRUB_s1_vkitti.pth",
    "bl_DELETE_s1_vkitti": "checkpoints/unlearned_bl_DELETE_s1_vkitti.pth",
    "TAE_vkitti": "checkpoints/unlearned_TAE_vkitti.pth",
    "PACE_vkitti": "checkpoints/unlearned_PACE_vkitti.pth",
}


def main():
    assert PERSON == CARCLS, "set FORGET_CLS=1 before running"
    device = "cuda"
    va = VKittiSeg("val", "orig", size=512, crop=False)
    va = Subset(va, list(range(min(4000, len(va)))))
    vl = DataLoader(va, batch_size=6, num_workers=8, pin_memory=True)

    out = {}
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} missing", flush=True)
            continue
        ck = torch.load(path, map_location=device, weights_only=False)
        m = get_seg_model(num_classes=NUM_CLASSES, device=device, arch=ARCH)
        m.load_state_dict(ck["state_dict"])
        r = seg_metrics(m, vl, device)
        out[name] = r
        print(f"{name:22s} car_IoU {r['car_IoU']:5.1f}  retain_mIoU {r['retain_mIoU']:5.1f}"
              f"  mIoU {r['mIoU']:5.1f}", flush=True)
        del m
        torch.cuda.empty_cache()

    os.makedirs("results/seg", exist_ok=True)
    json.dump(out, open("results/seg/utility_vkitti.json", "w"), indent=2)
    print("wrote results/seg/utility_vkitti.json")


if __name__ == "__main__":
    main()
