"""Single combined qualitative figure for the paper: two person rows on top,
two car rows below, sharing one column header. Replaces the two separate
per-class grids (saves a full figure of page budget) while keeping the point
legible: per-frame erasure looks complete for every unlearned model.

Usage: PYTHONPATH=. python scripts/make_qual_combined.py
"""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seg.model import get_seg_model, seg_forward

ARCH = "deeplabv3r50"
COLS = ["input", "ground truth", "original", "RL", "SCRUB", "SalUn", "PACE-T", "PACE (ours)"]

SETS = {
    "person": (60, {
        "original": f"checkpoints/vspw_original_{ARCH}_cls60.pth",
        "RL": "checkpoints/unlearned_bl_RL_s1.pth",
        "SCRUB": "checkpoints/unlearned_bl_SCRUB_s1_cls60.pth",
        "SalUn": "checkpoints/unlearned_bl_SalUn_s1.pth",
        "PACE-T": "checkpoints/unlearned_ADVR-T.pth",
        "PACE": "checkpoints/unlearned_ADVR-P.pth"}),
    "car": (48, {
        "original": f"checkpoints/vspw_original_{ARCH}_cls48.pth",
        "RL": "checkpoints/unlearned_bl_RL_s1_cls48.pth",
        "SCRUB": "checkpoints/unlearned_bl_SCRUB_s1_cls48.pth",
        "SalUn": "checkpoints/unlearned_bl_SalUn_s1_cls48.pth",
        "PACE-T": "checkpoints/unlearned_ADVR-T_car.pth",
        "PACE": "checkpoints/unlearned_ADVR-P_car.pth"}),
}

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


CROP = 0.58   # keep the central 58% of rows: the figure is about aspect ratio
              # on the page, and a landscape crop halves its height without
              # losing the object in any of the selected frames.


def vcrop(a):
    h = a.shape[0]; k = int(h * CROP); t = (h - k) // 2
    return a[t:t + k]


def denorm(t):
    return np.clip(t.permute(1, 2, 0).cpu().numpy() * STD + MEAN, 0, 1)


def overlay(img, mask, color, alpha=0.55):
    out = img.copy(); m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color)
    return out


def pick(ds, cls, n, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for i in rng.permutation(len(ds)):
        _, y = ds[int(i)]
        # require a reasonably large instance so it reads at print size
        if (y == cls).sum() > 4000:
            out.append(int(i))
        if len(out) >= n:
            break
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_per = 2
    fig, axes = plt.subplots(2 * n_per, len(COLS),
                             figsize=(9.2, 0.72 * 2 * n_per))

    row = 0
    for cls_name, (cls_idx, ckpts) in SETS.items():
        os.environ["FORGET_CLS"] = str(cls_idx)
        import importlib
        import seg.data as segdata
        importlib.reload(segdata)
        va = segdata.VSPWSeg("val", "orig", size=512, crop=False)
        idxs = pick(va, cls_idx, n_per, seed=1 if cls_name == "car" else 0)

        models = {}
        for k, p in ckpts.items():
            if not os.path.exists(p):
                print("[skip]", k, p); continue
            ck = torch.load(p, map_location=device, weights_only=False)
            m = get_seg_model(device=device, arch=ARCH)
            m.load_state_dict(ck["state_dict"]); m.eval()
            models[k] = m

        with torch.no_grad():
            for r, idx in enumerate(idxs):
                img_t, y = va[idx]
                img = denorm(img_t)
                axes[row, 0].imshow(vcrop(img))
                axes[row, 1].imshow(vcrop(overlay(img, y.numpy() == cls_idx, (0.85, 0.12, 0.12))))
                x = img_t.unsqueeze(0).to(device)
                for c, k in enumerate(ckpts.keys()):
                    ax = axes[row, 2 + c]
                    if k in models:
                        pred = seg_forward(models[k], x).argmax(1)[0].cpu().numpy()
                        ax.imshow(vcrop(overlay(img, pred == cls_idx, (0.10, 0.45, 0.90))))
                    else:
                        ax.imshow(vcrop(img))
                if r == 0:
                    axes[row, 0].set_ylabel(f"\\texttt{{{cls_name}}}", fontsize=8)
                row += 1
        for m in models.values():
            del m
        torch.cuda.empty_cache()

    for i in range(2 * n_per):
        for j in range(len(COLS)):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            for s in axes[i, j].spines.values():
                s.set_linewidth(0.4)
            if i == 0:
                axes[i, j].set_title(COLS[j], fontsize=8, pad=3)

    # class group labels down the left edge
    axes[0, 0].set_ylabel("person", fontsize=8.5)
    axes[n_per, 0].set_ylabel("car", fontsize=8.5)

    plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.03, right=0.995,
                        top=0.93, bottom=0.005)
    os.makedirs("paper/figures", exist_ok=True)
    p = "paper/figures/qualitative_combined.png"
    plt.savefig(p, dpi=260, bbox_inches="tight", pad_inches=0.02)
    print("wrote", p)


if __name__ == "__main__":
    main()
