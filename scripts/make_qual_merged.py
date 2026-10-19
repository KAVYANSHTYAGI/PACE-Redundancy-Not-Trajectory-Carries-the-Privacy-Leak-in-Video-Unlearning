"""Merged qualitative figure: replaces the separate own-prediction figure
(make_qual_combined.py) and attacker-reconstruction figure (make_qual_attack.py)
with one 4-row grid, all 8 columns shared:

  row 1: person example 1, each model's OWN prediction
  row 2: person example 2, each model's OWN prediction
  row 3: person, what the POOLED ATTACKER reconstructs (the payoff row --
         this is where per-frame-clean models visibly separate)
  row 4: car example, each model's OWN prediction

car has no attacker row: at this illustrative training budget the attacker
does not converge reliably on car (verified: even the undefended `original`
model stayed near-zero probability everywhere; see make_qual_attack.py's
docstring/history), so we do not show a misleading reconstruction for it.

Usage: PYTHONPATH=. python scripts/make_qual_merged.py
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Subset

from seg.attack import clip_output_field, train_temporal_attacker
from seg.model import get_seg_model, seg_forward

ARCH = "deeplabv3r50"
COLS = ["input", "ground truth", "original", "RL", "SCRUB", "SalUn", "PACE-T", "PACE (ours)"]
CLIP_LEN, FRAME_STRIDE, MIN_VIDEO_LEN, ACCUM, STEPS = 16, 2, 31, 4, 250

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
CROP = 0.58


def vcrop(a):
    h = a.shape[0]; k = int(h * CROP); t = (h - k) // 2
    return a[t:t + k]


def denorm(t):
    return np.clip(t.permute(1, 2, 0).cpu().numpy() * STD + MEAN, 0, 1)


def overlay(img, mask, color, alpha=0.55):
    out = img.copy(); m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color)
    return out


def heat_overlay(img, prob, lo, hi, alpha=0.75):
    cmap = matplotlib.colormaps["inferno"]
    norm = np.clip((prob - lo) / max(hi - lo, 1e-6), 0, 1)
    heat = cmap(norm)[..., :3]
    w = norm[..., None] * alpha
    return img * (1 - w) + heat * w


def pick(ds, cls, n, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for i in rng.permutation(len(ds)):
        _, y = ds[int(i)]
        if (y == cls).sum() > 4000:
            out.append(int(i))
        if len(out) >= n:
            break
    return out


def presence_filter(ds, cls_idx, min_px=800, cap=1200, top_n=150):
    """See make_qual_attack.py's version for the full rationale: a clip
    technically containing the class (at its anchor frame) can still be
    almost entirely negative supervision across a 16-frame/stride-2 window,
    which is what made an earlier unfiltered attempt fail to converge even
    for the undefended model. Ranks clips by the fraction of frames with
    real coverage instead of accept/reject, in one bounded pass."""
    n = min(cap, len(ds.clips))
    scored = []
    for i in range(n):
        hits = 0
        for _img_path, ann_path in ds.clips[i]:
            raw = np.array(Image.open(ann_path))
            lab = raw.astype(np.int64) - 1
            if int((lab == cls_idx).sum()) >= min_px:
                hits += 1
        frac = hits / len(ds.clips[i])
        if frac > 0:
            scored.append((frac, i))
    scored.sort(reverse=True)
    return [i for _f, i in scored[:top_n]]


def own_prediction_row(axes, row, cls_idx, ckpts, idx, va, device):
    img_t, y = va[idx]
    img = denorm(img_t)
    axes[row, 0].imshow(vcrop(img))
    axes[row, 1].imshow(vcrop(overlay(img, y.numpy() == cls_idx, (0.85, 0.12, 0.12))))
    x = img_t.unsqueeze(0).to(device)
    with torch.no_grad():
        for c, (name, path) in enumerate(ckpts.items()):
            ax = axes[row, 2 + c]
            if not os.path.exists(path):
                ax.imshow(vcrop(img)); continue
            ck = torch.load(path, map_location=device, weights_only=False)
            m = get_seg_model(device=device, arch=ARCH)
            m.load_state_dict(ck["state_dict"]); m.eval()
            pred = seg_forward(m, x).argmax(1)[0].cpu().numpy()
            ax.imshow(vcrop(overlay(img, pred == cls_idx, (0.10, 0.45, 0.90))))
            del m
    torch.cuda.empty_cache()


def attacker_row(axes, row, cls_name, cls_idx, ckpts, device):
    os.environ["FORGET_CLS"] = str(cls_idx)
    import importlib
    import seg.data as segdata
    importlib.reload(segdata)

    ctr_full = segdata.VSPWClipAnchored("train", size=512, clip_len=CLIP_LEN, crop=False,
                                        max_anchors=3000, seed=0,
                                        min_video_len=MIN_VIDEO_LEN, frame_stride=FRAME_STRIDE)
    cva = segdata.VSPWClipAnchored("val", size=512, clip_len=CLIP_LEN, crop=False,
                                   max_anchors=300, seed=0,
                                   min_video_len=MIN_VIDEO_LEN, frame_stride=FRAME_STRIDE)

    idx = presence_filter(ctr_full, cls_idx, min_px=800, cap=1200, top_n=200)
    print(f"[{cls_name}] well-covered train clips: {len(idx)}/{min(1200, len(ctr_full))} scanned",
          flush=True)
    ctr = Subset(ctr_full, idx) if idx else ctr_full
    ctl = DataLoader(ctr, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    cva_idx = presence_filter(cva, cls_idx, min_px=800, cap=300, top_n=100)
    if not cva_idx:
        cva_idx = list(range(len(cva)))
    ex_idx, t_show, best_score = None, None, -1.0
    for i in cva_idx:
        imgs_i, origs_i, _m = cva[i]
        per_frame = (origs_i == cls_idx).reshape(origs_i.shape[0], -1).sum(1)
        fmax = int(per_frame.max())
        if fmax < 8000:
            continue
        t = int(per_frame.argmax())
        mask = (origs_i[t] == cls_idx)
        ys, xs = mask.nonzero(as_tuple=True)
        H, W = mask.shape
        cy, cx = float(ys.float().mean()) / H, float(xs.float().mean()) / W
        centered = 0.15 < cx < 0.85 and 0.15 < cy < 0.85
        score = fmax * (1.0 if centered else 0.1)
        if score > best_score:
            ex_idx, t_show, best_score = i, t, score
    if ex_idx is None:
        ex_idx, t_show = 0, 0
    imgs, origs, _m = cva[ex_idx]
    imgs = imgs.unsqueeze(0).to(device)
    origs = origs.unsqueeze(0).to(device)

    img_show = denorm(imgs[0, t_show].cpu())
    axes[row, 0].imshow(vcrop(img_show))
    gt_mask = (origs[0, t_show].cpu().numpy() == cls_idx)
    axes[row, 1].imshow(vcrop(overlay(img_show, gt_mask, (0.85, 0.12, 0.12))))

    prob_maps = {}
    for name, path in ckpts.items():
        if not os.path.exists(path):
            print("[skip]", name, path); continue
        ck = torch.load(path, map_location=device, weights_only=False)
        model = get_seg_model(device=device, arch=ARCH)
        model.load_state_dict(ck["state_dict"]); model.eval()

        D = train_temporal_attacker(model, ctl, device, epochs=10 ** 6,
                                    max_iters=STEPS, kind="mamba", accum_steps=ACCUM)
        D.eval()
        with torch.no_grad():
            O = clip_output_field(model, imgs, out_size=128)
            prob = torch.sigmoid(D(O))[0, t_show].cpu().numpy()
        prob_up = F.interpolate(torch.from_numpy(prob)[None, None],
                                size=img_show.shape[:2], mode="bilinear",
                                align_corners=False)[0, 0].numpy()
        prob_maps[name] = prob_up
        print(f"[{cls_name}] {name}: prob min {prob_up.min():.3f} max {prob_up.max():.3f} "
              f"mean {prob_up.mean():.3f}", flush=True)
        del model, D
        torch.cuda.empty_cache()

    stacked = np.stack(list(prob_maps.values())) if prob_maps else np.zeros((1, 1, 1))
    lo, hi = np.percentile(stacked, 2), np.percentile(stacked, 98)
    for c, name in enumerate(ckpts.keys()):
        ax = axes[row, 2 + c]
        if name not in prob_maps:
            ax.imshow(vcrop(img_show)); continue
        ax.imshow(vcrop(heat_overlay(img_show, prob_maps[name], lo, hi)))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fig, axes = plt.subplots(4, len(COLS), figsize=(9.2, 0.72 * 4))

    os.environ["FORGET_CLS"] = str(SETS["person"][0])
    import importlib
    import seg.data as segdata
    importlib.reload(segdata)
    va_person = segdata.VSPWSeg("val", "orig", size=512, crop=False)
    person_idxs = pick(va_person, SETS["person"][0], 2, seed=0)
    own_prediction_row(axes, 0, SETS["person"][0], SETS["person"][1], person_idxs[0], va_person, device)
    own_prediction_row(axes, 1, SETS["person"][0], SETS["person"][1], person_idxs[1], va_person, device)

    attacker_row(axes, 2, "person", SETS["person"][0], SETS["person"][1], device)

    os.environ["FORGET_CLS"] = str(SETS["car"][0])
    importlib.reload(segdata)
    va_car = segdata.VSPWSeg("val", "orig", size=512, crop=False)
    car_idx = pick(va_car, SETS["car"][0], 1, seed=1)[0]
    own_prediction_row(axes, 3, SETS["car"][0], SETS["car"][1], car_idx, va_car, device)

    for i in range(4):
        for j in range(len(COLS)):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            for s in axes[i, j].spines.values():
                s.set_linewidth(0.4)
            if i == 0:
                axes[i, j].set_title(COLS[j], fontsize=8, pad=3)

    axes[0, 0].set_ylabel("person", fontsize=8.5)
    axes[2, 0].set_ylabel("attack", fontsize=8.5)
    axes[3, 0].set_ylabel("car", fontsize=8.5)

    plt.subplots_adjust(wspace=0.02, hspace=0.03, left=0.03, right=0.995,
                        top=0.94, bottom=0.005)
    os.makedirs("paper/figures", exist_ok=True)
    p = "paper/figures/qualitative_merged.png"
    plt.savefig(p, dpi=260, bbox_inches="tight", pad_inches=0.02)
    print("wrote", p)


if __name__ == "__main__":
    main()
