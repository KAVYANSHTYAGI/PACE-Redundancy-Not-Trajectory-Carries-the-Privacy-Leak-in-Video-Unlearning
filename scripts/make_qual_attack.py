"""Qualitative ATTACKER RECONSTRUCTION figure -- the complement to
make_qual_combined.py.

The existing qualitative figure (Fig. 4) shows each model's own per-frame
prediction, which looks clean for every method including the ones Tables 3/4
score as LEAK -- that is the whole point of the paper (per-frame audits are
blind to pooled leakage) but it means Fig. 4 cannot visually distinguish PACE
from a leaky baseline. This script instead trains the SAME pooled/temporal
attacker used in the main protocol (mamba, T=16, stride=2) against each
model's output field and renders what the attacker RECONSTRUCTS -- which is
exactly the quantity Tables 3/4/5 report as pooled AP, and is where the
methods visibly separate.

This is illustrative (one training run per model, reduced step budget), not a
re-measurement -- Tables 3/4/5 remain the source of truth for the numbers.

Usage: PYTHONPATH=. python scripts/make_qual_attack.py
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

from seg.attack import clip_output_field, clip_person_mask_low, train_temporal_attacker
from seg.model import get_seg_model

ARCH = "deeplabv3r50"
CLIP_LEN = 16
FRAME_STRIDE = 2
MIN_VIDEO_LEN = 31
ACCUM = 4
# car has far fewer eligible training clips than person (see EXPERIMENTS.md),
# so its attacker needs more steps to reach comparable convergence at this
# illustrative budget; person already converges cleanly at 250 (validated:
# original/RL land at high mean confidence, PACE-T/PACE far below).
STEPS_BY_CLASS = {"person": 250, "car": 600}
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


def presence_filter(ds, cls_idx, min_px=800, cap=1200, top_n=150):
    """Indices into ds.clips (a VSPWClipAnchored), ranked by what fraction of
    the clip's frames have >=min_px pixels of cls_idx, keeping the top_n.

    VSPWClipAnchored only guarantees the class at the ANCHOR (centre) frame;
    at T=16/stride=2 (~2.1s span) a car can drift out of frame across most of
    the window, so a clip technically "containing car" can still be almost
    entirely negative supervision -- which is why the attacker failed to
    converge even at 600 steps on an unfiltered clip pool (verified: even the
    undefended `original` model stayed at ~0 probability everywhere). Single
    bounded pass (cap clips, not the whole pool) keeps this a fixed, small
    cost regardless of dataset size; reads only the annotation PNGs (no RGB
    decode)."""
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


def heat_overlay(img, prob, lo, hi, alpha=0.75):
    """Blend a probability map onto img using an inferno colormap, rescaled
    by a (lo, hi) pair SHARED across every model shown for one example.

    Raw sigmoid outputs are not comparable across models this way: AP is a
    threshold-free, ranking-based metric (Sec. 5.4 uses the same reasoning
    for scoring), so a converged attacker's absolute confidence can sit well
    below 0.5 even when it separates the class cleanly in rank. Normalising
    each model's heatmap independently would equalise that -- a uniformly
    unconfident attacker and a sharply peaked one would render identically
    bright -- which overstates leakage exactly where PACE succeeds. Sharing
    one (lo, hi) across all models for the same example instead lets a
    genuinely weaker signal render visibly weaker."""
    cmap = matplotlib.colormaps["inferno"]
    norm = np.clip((prob - lo) / max(hi - lo, 1e-6), 0, 1)
    heat = cmap(norm)[..., :3]
    w = norm[..., None] * alpha
    return img * (1 - w) + heat * w


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fig, axes = plt.subplots(2, len(COLS), figsize=(9.2, 0.72 * 2 * 1.9))

    for row, (cls_name, (cls_idx, ckpts)) in enumerate(SETS.items()):
        steps = STEPS_BY_CLASS[cls_name]
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

        # Require the class to persist across most of the clip, not just the
        # anchor frame -- see presence_filter's docstring for why this is the
        # actual fix (the earlier failure was not a step-count problem).
        idx = presence_filter(ctr_full, cls_idx, min_px=800, cap=1200, top_n=200)
        print(f"[{cls_name}] well-covered train clips: {len(idx)}/{min(1200, len(ctr_full))} scanned",
              flush=True)
        ctr = Subset(ctr_full, idx) if idx else ctr_full
        ctl = DataLoader(ctr, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

        # one fixed example clip, reused across every model for a fair comparison.
        # Restrict the search to well-covered clips (same standard as the
        # training pool) so the displayed example is representative of what
        # the attacker actually trained on, not a lucky/unlucky one-off.
        # Within that pool, pick the clip AND displayed frame together, scoring
        # every candidate for a large, roughly centred instance -- a
        # technically-passing but tiny or corner-cropped instance renders as
        # an unreadable example.
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

        # Pass 1: train each model's attacker and collect its raw probability
        # map. Rendering is deferred to pass 2 so every column in this row can
        # share one normalisation scale (see heat_overlay's docstring).
        prob_maps = {}
        for name, path in ckpts.items():
            if not os.path.exists(path):
                print("[skip]", name, path); continue
            ck = torch.load(path, map_location=device, weights_only=False)
            model = get_seg_model(device=device, arch=ARCH)
            model.load_state_dict(ck["state_dict"]); model.eval()

            D = train_temporal_attacker(model, ctl, device, epochs=10 ** 6,
                                        max_iters=steps, kind="mamba",
                                        accum_steps=ACCUM)
            D.eval()
            with torch.no_grad():
                O = clip_output_field(model, imgs, out_size=128)
                prob = torch.sigmoid(D(O))[0, t_show].cpu().numpy()  # [h,w] in [0,1]
            prob_up = F.interpolate(torch.from_numpy(prob)[None, None],
                                    size=img_show.shape[:2], mode="bilinear",
                                    align_corners=False)[0, 0].numpy()
            prob_maps[name] = prob_up
            print(f"[{cls_name}] {name}: prob min {prob_up.min():.3f} "
                  f"max {prob_up.max():.3f} mean {prob_up.mean():.3f}", flush=True)

            del model, D
            torch.cuda.empty_cache()

        # Pass 2: shared scale from every rendered model's 2nd/98th percentile
        # (robust to a single outlier pixel), then render.
        stacked = np.stack(list(prob_maps.values())) if prob_maps else np.zeros((1, 1, 1))
        lo, hi = np.percentile(stacked, 2), np.percentile(stacked, 98)
        for c, name in enumerate(ckpts.keys()):
            ax = axes[row, 2 + c]
            if name not in prob_maps:
                ax.imshow(vcrop(img_show)); continue
            ax.imshow(vcrop(heat_overlay(img_show, prob_maps[name], lo, hi)))

        axes[row, 0].set_ylabel(f"\\texttt{{{cls_name}}}", fontsize=8.5)

    for i in range(2):
        for j in range(len(COLS)):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            for s in axes[i, j].spines.values():
                s.set_linewidth(0.4)
            if i == 0:
                axes[i, j].set_title(COLS[j], fontsize=8, pad=3)

    plt.subplots_adjust(wspace=0.02, hspace=0.03, left=0.03, right=0.995,
                        top=0.92, bottom=0.01)
    os.makedirs("paper/figures", exist_ok=True)
    p = "paper/figures/qualitative_attack.png"
    plt.savefig(p, dpi=260, bbox_inches="tight", pad_inches=0.02)
    print("wrote", p)


if __name__ == "__main__":
    main()
