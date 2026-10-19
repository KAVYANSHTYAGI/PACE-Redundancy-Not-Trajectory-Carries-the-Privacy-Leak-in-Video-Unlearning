"""Fast utility check (person_IoU / retain_mIoU / mIoU) across all person-class
checkpoints, no relearn audit -- just seg_metrics on a capped val set. Mirrors
scripts/utility_check_car.py; needed to fill Table 3's utility column for the
newly-run 8 published baselines on person.

Usage: PYTHONPATH=. python scripts/utility_check_person.py
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
    "ADVR": "checkpoints/unlearned_ADVR.pth",
    "ADVR-T": "checkpoints/unlearned_ADVR-T.pth",
    "bl_FT_s1": "checkpoints/unlearned_bl_FT_s1.pth",
    "bl_NegGrad_s1_cls60": "checkpoints/unlearned_bl_NegGrad_s1_cls60.pth",
    "bl_NegGrad+_s1": "checkpoints/unlearned_bl_NegGrad+_s1.pth",
    "bl_RL_s1": "checkpoints/unlearned_bl_RL_s1.pth",
    "bl_SalUn_s1": "checkpoints/unlearned_bl_SalUn_s1.pth",
    "bl_SSD_s1": "checkpoints/unlearned_bl_SSD_s1.pth",
    "bl_SCRUB_s1_cls60": "checkpoints/unlearned_bl_SCRUB_s1_cls60.pth",
    "bl_DELETE_s1": "checkpoints/unlearned_bl_DELETE_s1.pth",
    "ADVR-P": "checkpoints/unlearned_ADVR-P.pth",
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
        print(f"{name:22s} person_IoU {r['person_IoU']:5.1f}  retain_mIoU {r['retain_mIoU']:5.1f}"
              f"  mIoU {r['mIoU']:5.1f}", flush=True)
        del m
        torch.cuda.empty_cache()

    os.makedirs("results/seg", exist_ok=True)
    json.dump(out, open("results/seg/utility_person.json", "w"), indent=2)
    print("wrote results/seg/utility_person.json")


if __name__ == "__main__":
    main()
