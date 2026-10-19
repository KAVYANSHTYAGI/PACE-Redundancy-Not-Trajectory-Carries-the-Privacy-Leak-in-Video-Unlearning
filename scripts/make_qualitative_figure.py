"""Qualitative figure generator for the paper: for a handful of val frames
containing the forget class, render input | ground truth | prediction overlay
for a small model set (original, one representative baseline, ADVR-T, ADVR-P),
so a reader can *see* the unlearning effect rather than only read AP numbers.

One row per sample frame, one column per model (+ input, +GT). Saved as a
single PNG per forget class (person, car) under paper/figures/.

Usage:
  PYTHONPATH=. python scripts/make_qualitative_figure.py --cls person --n 4
  PYTHONPATH=. FORGET_CLS=48 python scripts/make_qualitative_figure.py --cls car --n 4
"""
import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from seg.data import VSPWSeg, PERSON
from seg.model import get_seg_model, seg_forward

ARCH = "deeplabv3r50"

# forget-class-specific checkpoint sets. "baseline" is one representative
# published method (RL -- a common reference point, not the best/worst) shown
# alongside the two method-line variants, so the figure stays legible instead
# of showing all 8 baselines.
MODEL_SETS = {
    "person": {
        "original": f"checkpoints/vspw_original_{ARCH}_cls60.pth",
        "bl_RL": "checkpoints/unlearned_bl_RL_s1.pth",
        "ADVR-T": "checkpoints/unlearned_ADVR-T.pth",
        "ADVR-P": "checkpoints/unlearned_ADVR-P.pth",
    },
    "car": {
        "original": f"checkpoints/vspw_original_{ARCH}_cls48.pth",
        "bl_RL": "checkpoints/unlearned_bl_RL_s1_cls48.pth",
        "ADVR-T": "checkpoints/unlearned_ADVR-T_car.pth",
        "ADVR-P": "checkpoints/unlearned_ADVR-P_car.pth",
    },
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denorm(img_t):
    img = img_t.permute(1, 2, 0).cpu().numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def overlay(img, mask, color=(1.0, 0.0, 0.0), alpha=0.5):
    out = img.copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color)
    return out


def pick_frames(ds, forget_cls, n, seed=0):
    """Scan for `n` frames whose label contains forget_cls, spread across the
    dataset rather than clustered in one video."""
    rng = np.random.RandomState(seed)
    idxs = rng.permutation(len(ds))
    picked = []
    for i in idxs:
        _, y = ds[int(i)]
        if (y == forget_cls).any():
            picked.append(int(i))
        if len(picked) >= n:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", choices=["person", "car"], required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    forget_cls = PERSON  # env-driven: set FORGET_CLS=48 before running for car
    models = MODEL_SETS[args.cls]

    va = VSPWSeg("val", "orig", size=512, crop=False)
    frame_idxs = pick_frames(va, forget_cls, args.n)
    if not frame_idxs:
        raise SystemExit(f"no val frames found containing class {forget_cls} "
                          f"-- did you forget to set FORGET_CLS=48 for --cls car?")

    loaded = {}
    for name, path in models.items():
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} missing")
            continue
        ck = torch.load(path, map_location=device, weights_only=False)
        m = get_seg_model(device=device, arch=ARCH)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        loaded[name] = m

    cols = ["input", "ground truth"] + list(loaded.keys())
    n_rows, n_cols = len(frame_idxs), len(cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for r, idx in enumerate(frame_idxs):
            img_t, y = va[idx]
            img = denorm(img_t)
            gt_mask = (y.numpy() == forget_cls)

            axes[r, 0].imshow(img)
            axes[r, 1].imshow(overlay(img, gt_mask))

            x = img_t.unsqueeze(0).to(device)
            for c, name in enumerate(loaded.keys()):
                logits = seg_forward(loaded[name], x)
                pred = logits.argmax(1)[0].cpu().numpy()
                pred_mask = (pred == forget_cls)
                axes[r, 2 + c].imshow(overlay(img, pred_mask, color=(0.0, 0.6, 1.0)))

            for c in range(n_cols):
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
                if r == 0:
                    axes[r, c].set_title(cols[c], fontsize=11)

    plt.tight_layout()
    out = args.out or f"paper/figures/qualitative_{args.cls}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
