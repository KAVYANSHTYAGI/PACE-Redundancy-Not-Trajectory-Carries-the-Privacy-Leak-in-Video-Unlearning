"""Multi-example attacker-reconstruction figure, PAIRED with own-predictions.

Why this exists. make_qual_attack.py / make_qual_merged.py each trained six
attackers (~20 min) and then rendered exactly ONE example frame from them,
which is backwards: attacker TRAINING is the expensive part, while evaluating
an already-trained attacker on another val clip costs milliseconds. Here the
loop is inverted -- outer loop over models (expensive, once each), inner loop
over examples (free) -- so one training run yields several examples instead of
one.

Layout, per example: an own-prediction row immediately above its attack row,
on the SAME frame, so the contrast is direct rather than something the reader
has to hold across figures. Every method looks clean in the prediction row;
only the attack row separates them.

Examples are chosen deterministically (top-scoring by forget-class coverage
and centring, deduplicated by source video) BEFORE any attacker is trained, so
this is not cherry-picking runs that happened to flatter PACE.

Usage: PYTHONPATH=. python scripts/make_qual_attack_multi.py
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
N_EXAMPLES = 3
PERSON = 60
CKPTS = {
    "original": f"checkpoints/vspw_original_{ARCH}_cls60.pth",
    "RL": "checkpoints/unlearned_bl_RL_s1.pth",
    "SCRUB": "checkpoints/unlearned_bl_SCRUB_s1_cls60.pth",
    "SalUn": "checkpoints/unlearned_bl_SalUn_s1.pth",
    "PACE-T": "checkpoints/unlearned_ADVR-T.pth",
    "PACE": "checkpoints/unlearned_ADVR-P.pth",
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
    """Shared (lo, hi) across every model shown for one example -- see
    make_qual_attack.py: normalising each model independently would render a
    uniformly unconfident attacker as brightly as a sharply peaked one, which
    overstates leakage exactly where PACE succeeds."""
    cmap = matplotlib.colormaps["inferno"]
    norm = np.clip((prob - lo) / max(hi - lo, 1e-6), 0, 1)
    heat = cmap(norm)[..., :3]
    w = norm[..., None] * alpha
    return img * (1 - w) + heat * w


def presence_filter(ds, cls_idx, min_px=800, cap=1200, top_n=150):
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


def select_examples(cva, cls_idx, n):
    """Top-n (clip, frame) pairs by instance size and centring, at most one per
    source video so the rows are visually distinct rather than n neighbouring
    frames of the same shot."""
    cand_idx = presence_filter(cva, cls_idx, min_px=800, cap=300, top_n=100)
    if not cand_idx:
        cand_idx = list(range(len(cva)))
    scored = []
    for i in cand_idx:
        _imgs_i, origs_i, _m = cva[i]
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
        scored.append((fmax * (1.0 if centered else 0.1), i, t))
    scored.sort(reverse=True)

    out, seen_videos = [], set()
    for _s, i, t in scored:
        vid = cva.anchors[i][0] if hasattr(cva, "anchors") else i
        if vid in seen_videos:
            continue
        seen_videos.add(vid)
        out.append((i, t))
        if len(out) >= n:
            break
    while len(out) < n and scored:            # fall back if too few videos
        _s, i, t = scored[len(out) % len(scored)]
        out.append((i, t))
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["FORGET_CLS"] = str(PERSON)
    import importlib
    import seg.data as segdata
    importlib.reload(segdata)

    ctr_full = segdata.VSPWClipAnchored("train", size=512, clip_len=CLIP_LEN, crop=False,
                                        max_anchors=3000, seed=0,
                                        min_video_len=MIN_VIDEO_LEN, frame_stride=FRAME_STRIDE)
    cva = segdata.VSPWClipAnchored("val", size=512, clip_len=CLIP_LEN, crop=False,
                                   max_anchors=300, seed=0,
                                   min_video_len=MIN_VIDEO_LEN, frame_stride=FRAME_STRIDE)

    tr_idx = presence_filter(ctr_full, PERSON, min_px=800, cap=1200, top_n=200)
    print(f"well-covered train clips: {len(tr_idx)}/{min(1200, len(ctr_full))} scanned", flush=True)
    ctr = Subset(ctr_full, tr_idx) if tr_idx else ctr_full
    ctl = DataLoader(ctr, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    examples = select_examples(cva, PERSON, N_EXAMPLES)
    print("examples (clip, frame):", examples, flush=True)
    ex_data = []
    for i, t in examples:
        imgs, origs, _m = cva[i]
        ex_data.append((imgs.unsqueeze(0).to(device), origs.unsqueeze(0).to(device), t))

    # Outer loop over models (expensive: one attacker training each), inner
    # loop over examples (free: the attacker is already trained).
    preds, probs = {}, {}
    for name, path in CKPTS.items():
        if not os.path.exists(path):
            print("[skip]", name, path, flush=True); continue
        ck = torch.load(path, map_location=device, weights_only=False)
        model = get_seg_model(device=device, arch=ARCH)
        model.load_state_dict(ck["state_dict"]); model.eval()

        D = train_temporal_attacker(model, ctl, device, epochs=10 ** 6,
                                    max_iters=STEPS, kind="mamba", accum_steps=ACCUM)
        D.eval()
        with torch.no_grad():
            for e, (imgs, _origs, t_show) in enumerate(ex_data):
                preds[(name, e)] = seg_forward(
                    model, imgs[:, t_show]).argmax(1)[0].cpu().numpy()
                O = clip_output_field(model, imgs, out_size=128)
                prob = torch.sigmoid(D(O))[0, t_show].cpu().numpy()
                hw = imgs.shape[-2:]
                probs[(name, e)] = F.interpolate(
                    torch.from_numpy(prob)[None, None], size=hw,
                    mode="bilinear", align_corners=False)[0, 0].numpy()
                print(f"  {name} ex{e}: prob max {probs[(name, e)].max():.3f} "
                      f"mean {probs[(name, e)].mean():.3f}", flush=True)
        del model, D
        torch.cuda.empty_cache()

    nrows = 2 * N_EXAMPLES
    fig, axes = plt.subplots(nrows, len(COLS), figsize=(9.2, 0.72 * nrows))
    for e, (imgs, origs, t_show) in enumerate(ex_data):
        img = denorm(imgs[0, t_show].cpu())
        gt = (origs[0, t_show].cpu().numpy() == PERSON)
        r_pred, r_att = 2 * e, 2 * e + 1

        for r in (r_pred, r_att):
            axes[r, 0].imshow(vcrop(img))
            axes[r, 1].imshow(vcrop(overlay(img, gt, (0.85, 0.12, 0.12))))

        for c, name in enumerate(CKPTS):
            if (name, e) in preds:
                axes[r_pred, 2 + c].imshow(
                    vcrop(overlay(img, preds[(name, e)] == PERSON, (0.10, 0.45, 0.90))))
            else:
                axes[r_pred, 2 + c].imshow(vcrop(img))

        stack = np.stack([probs[(n, e)] for n in CKPTS if (n, e) in probs])
        lo, hi = np.percentile(stack, 2), np.percentile(stack, 98)
        for c, name in enumerate(CKPTS):
            if (name, e) in probs:
                axes[r_att, 2 + c].imshow(vcrop(heat_overlay(img, probs[(name, e)], lo, hi)))
            else:
                axes[r_att, 2 + c].imshow(vcrop(img))

        axes[r_pred, 0].set_ylabel("predicted", fontsize=7.5)
        axes[r_att, 0].set_ylabel("recovered", fontsize=7.5)

    for i in range(nrows):
        for j in range(len(COLS)):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            for s in axes[i, j].spines.values():
                s.set_linewidth(0.4)
            if i == 0:
                axes[i, j].set_title(COLS[j], fontsize=8, pad=3)

    plt.subplots_adjust(wspace=0.02, hspace=0.03, left=0.035, right=0.995,
                        top=0.95, bottom=0.005)
    os.makedirs("paper/figures", exist_ok=True)
    p = "paper/figures/qualitative_attack_multi.png"
    plt.savefig(p, dpi=260, bbox_inches="tight", pad_inches=0.02)
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
