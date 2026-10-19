"""Cut the two strip videos in docs/assets/videos into per-method sprite sheets
for the interactive viewer on the project page.

Each strip frame holds eight tiles side by side (input, ground truth, original,
RL, SCRUB, SalUn, PACE-T, PACE). This script extracts the 16 frames, crops each
tile, upscales it 2x, and packs the 16 frames of one tile into a 4x4 sprite
sheet at docs/assets/frames/<kind>_<name>.jpg.

It also writes docs/assets/videos/pano_*.mp4, the same strips cropped to the row
of tiles and their column titles (dropping the blank band, the frame counter and
the 1 px border the renderer leaves on the bottom and right edge), for the
full-width looping strip on the page.

Usage: python scripts/make_page_assets.py
Requires ffmpeg on PATH and Pillow.
"""
import os
import subprocess
import tempfile

from PIL import Image

VID_DIR = "docs/assets/videos"
OUT_DIR = "docs/assets/frames"
NAMES = ["input", "gt", "original", "RL", "SCRUB", "SalUn", "PACE-T", "PACE"]
X0 = [17, 258, 499, 741, 982, 1223, 1464, 1705]  # measured tile left edges in the 1938x286 frame
TILE_W, Y0, TILE_H = 216, 143, 125
SCALE, COLS, N = 2, 4, 16


def make_pano(src, dst):
    # keep x 0..1935 and y 108..269: column titles and tiles only (even sizes for yuv420p)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", "crop=1936:162:0:108",
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst], check=True)


def frames(mp4):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["ffmpeg", "-v", "error", "-i", mp4, os.path.join(d, "f%03d.png")], check=True)
        return [Image.open(os.path.join(d, f"f{i:03d}.png")).convert("RGB") for i in range(1, N + 1)]


def sheet(tiles):
    w, h = TILE_W * SCALE, TILE_H * SCALE
    out = Image.new("RGB", (w * COLS, h * (N // COLS)), "white")
    for i, t in enumerate(tiles):
        t = t.resize((w, h), Image.LANCZOS)
        out.paste(t, ((i % COLS) * w, (i // COLS) * h))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for kind, mp4 in [("pred", "predictions.mp4"), ("att", "attack_recovery.mp4")]:
        make_pano(os.path.join(VID_DIR, mp4), os.path.join(VID_DIR, "pano_" + mp4))
        fr = frames(os.path.join(VID_DIR, mp4))
        assert len(fr) == N and fr[0].size == (1938, 286), (len(fr), fr[0].size)
        for name, x in zip(NAMES, X0):
            if kind == "att" and name in ("input", "gt"):
                continue  # identical to the predictions video
            tiles = [f.crop((x, Y0, x + TILE_W, Y0 + TILE_H)) for f in fr]
            sheet(tiles).save(os.path.join(OUT_DIR, f"{kind}_{name}.jpg"), quality=88)
    print("wrote", sorted(os.listdir(OUT_DIR)))


if __name__ == "__main__":
    main()
