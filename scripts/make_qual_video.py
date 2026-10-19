"""Video-form qualitative demo: renders EVERY frame of a real VSPW clip (not
just the single best frame make_qual_attack_multi.py picks), for two panels:

  1. predictions.mp4  -- input | ground truth | each method's own per-frame
     prediction, animated across the clip. This is what a per-frame audit
     sees: every method looks clean here.
  2. attack_recovery.mp4 -- the same layout, but each method's column is the
     POOLED ATTACKER's recovered-probability heatmap instead of the model's
     own prediction. This is the paper's actual point made visible in motion:
     baselines that look safe in (1) visibly reveal the forgotten class's
     silhouette in (2), while PACE stays flat.

Frame panels are also encoded as looping GIFs (same content, smaller/slower)
since GitHub renders GIFs inline in Markdown but not repo-relative <video>
tags reliably outside a Pages site.

Usage: PYTHONPATH=. python scripts/make_qual_video.py
Requires ffmpeg on PATH. Reuses the exact example-selection and attacker-
training logic from make_qual_attack_multi.py so the clip shown here is the
same one (or same family) already vetted for the paper's static figure.
"""
import os
import shutil
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from seg.attack import clip_output_field, train_temporal_attacker
from seg.model import get_seg_model, seg_forward
from scripts.make_qual_attack_multi import presence_filter, select_examples, denorm, vcrop, overlay, heat_overlay

ARCH = "deeplabv3r50"
COLS = ["input", "ground truth", "original", "RL", "SCRUB", "SalUn", "PACE-T", "PACE (ours)"]
CLIP_LEN, FRAME_STRIDE, MIN_VIDEO_LEN, ACCUM, STEPS = 16, 2, 31, 4, 250
PERSON = 60
CKPTS = {
    "original": f"checkpoints/vspw_original_{ARCH}_cls60.pth",
    "RL": "checkpoints/unlearned_bl_RL_s1.pth",
    "SCRUB": "checkpoints/unlearned_bl_SCRUB_s1_cls60.pth",
    "SalUn": "checkpoints/unlearned_bl_SalUn_s1.pth",
    "PACE-T": "checkpoints/unlearned_ADVR-T.pth",
    "PACE": "checkpoints/unlearned_ADVR-P.pth",
}
OUT_DIR = "paper/videos"
FRAME_DIR = "/tmp/pace_qual_video_frames"
FPS = 4  # clip spans 2.1s of real time at 16 frames/stride-2 -- slow enough to actually see, not a blur


def render_panel(fig_axes_builder, out_name):
    frames_dir = os.path.join(FRAME_DIR, out_name)
    os.makedirs(frames_dir, exist_ok=True)
    return frames_dir


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FRAME_DIR, exist_ok=True)
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
    ctr = Subset(ctr_full, tr_idx) if tr_idx else ctr_full
    ctl = DataLoader(ctr, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    # one clip, chosen by the same "best example" heuristic as the paper figure
    (clip_idx, _best_t), = select_examples(cva, PERSON, 1)
    imgs, origs, _m = cva[clip_idx]
    imgs = imgs.unsqueeze(0).to(device)      # [1, T, 3, H, W]
    origs = origs.unsqueeze(0).to(device)    # [1, T, H, W]
    T = imgs.shape[1]
    print(f"rendering clip {clip_idx}, {T} frames", flush=True)

    preds_all = {}   # name -> [T, H, W] bool prediction
    probs_all = {}   # name -> [T, H, W] float recovered-probability
    for name, path in CKPTS.items():
        if not os.path.exists(path):
            print("[skip]", name, path, flush=True)
            continue
        ck = torch.load(path, map_location=device, weights_only=False)
        model = get_seg_model(device=device, arch=ARCH)
        model.load_state_dict(ck["state_dict"])
        model.eval()

        D = train_temporal_attacker(model, ctl, device, epochs=10 ** 6,
                                    max_iters=STEPS, kind="mamba", accum_steps=ACCUM)
        D.eval()
        with torch.no_grad():
            pred_t = []
            for t in range(T):
                pred_t.append(seg_forward(model, imgs[:, t]).argmax(1)[0].cpu().numpy())
            preds_all[name] = np.stack(pred_t)

            O = clip_output_field(model, imgs, out_size=128)
            prob = torch.sigmoid(D(O))[0].cpu().numpy()  # [T, 128, 128]
            hw = imgs.shape[-2:]
            prob_up = F.interpolate(torch.from_numpy(prob)[:, None], size=hw,
                                     mode="bilinear", align_corners=False)[:, 0].numpy()
            probs_all[name] = prob_up
        print(f"  {name}: mean recovered prob {probs_all[name].mean():.3f}, "
              f"max {probs_all[name].max():.3f}", flush=True)
        del model, D
        torch.cuda.empty_cache()

    stack = np.stack([probs_all[n] for n in CKPTS if n in probs_all])
    lo, hi = np.percentile(stack, 2), np.percentile(stack, 98)

    def build_and_encode(mode: str, out_name: str):
        frames_dir = os.path.join(FRAME_DIR, out_name)
        os.makedirs(frames_dir, exist_ok=True)
        for t in range(T):
            img = denorm(imgs[0, t].cpu())
            gt = (origs[0, t].cpu().numpy() == PERSON)
            fig, axes = plt.subplots(1, len(COLS), figsize=(11.5, 1.9))
            axes[0].imshow(vcrop(img))
            axes[1].imshow(vcrop(overlay(img, gt, (0.85, 0.12, 0.12))))
            for c, name in enumerate(CKPTS):
                ax = axes[2 + c]
                if name not in preds_all:
                    ax.imshow(vcrop(img))
                elif mode == "pred":
                    ax.imshow(vcrop(overlay(img, preds_all[name][t] == PERSON, (0.10, 0.45, 0.90))))
                else:
                    ax.imshow(vcrop(heat_overlay(img, probs_all[name][t], lo, hi)))
            for c, title in enumerate(COLS):
                axes[c].set_title(title, fontsize=7.5)
                axes[c].axis("off")
            fig.suptitle(f"frame {t + 1}/{T}", fontsize=8, y=1.02)
            fig.tight_layout()
            fig.savefig(os.path.join(frames_dir, f"f{t:03d}.png"), dpi=170, bbox_inches="tight")
            plt.close(fig)

        mp4_path = os.path.join(OUT_DIR, f"{out_name}.mp4")
        gif_path = os.path.join(OUT_DIR, f"{out_name}.gif")
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(FPS), "-i", os.path.join(frames_dir, "f%03d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4_path,
        ], check=True, capture_output=True)
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(FPS), "-i", os.path.join(frames_dir, "f%03d.png"),
            "-vf", "fps=4,scale=900:-1:flags=lanczos,split[s0][s1];"
                   "[s0]palettegen[p];[s1][p]paletteuse",
            "-loop", "0", gif_path,
        ], check=True, capture_output=True)
        print(f"wrote {mp4_path} and {gif_path}", flush=True)

    build_and_encode("pred", "predictions")
    build_and_encode("attack", "attack_recovery")
    shutil.rmtree(FRAME_DIR, ignore_errors=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
