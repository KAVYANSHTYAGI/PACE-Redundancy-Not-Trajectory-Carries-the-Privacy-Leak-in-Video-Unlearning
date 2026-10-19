"""Virtual KITTI 2 dataset loader, mirroring seg/data.py's VSPW conventions
(VSPWSeg / VSPWUnlearn / VSPWClip / VSPWClipAnchored) so the existing
unlearning pipeline (baselines, FAE/TAE/PACE, run_temporal_protocol.py) can
run against a second dataset with minimal changes elsewhere.

Layout on disk (data/VKITTI2/):
  Scene{01,02,06,18,20}/{clone,fog,morning,overcast,rain,sunset,
                          15-deg-left,15-deg-right,30-deg-left,30-deg-right}/
    frames/rgb/Camera_{0,1}/rgb_%05d.jpg
    frames/classSegmentation/Camera_{0,1}/classgt_%05d.png

Every frame is densely labeled (fully synthetic ground truth) -- unlike
CamVid's sparse stills, this is what makes clip-based temporal evaluation
valid here. A "video" in this dataset's sense is one (scene, variant, camera)
sequence, played in frame-index order, exactly analogous to one VSPW video_id.

Class-color palette confirmed against extracted classSegmentation PNGs: 11/15
official colors observed in a sampled frame, all exact matches; a broader
28-image scan across every scene/camera found zero unrecognized colors.

Forget class: CAR (rigid, VKITTI2's synthetic vehicles), the direct analog of
the VSPW car-class generalization study -- this makes person/car (VSPW) vs car
(VKITTI2) a genuine cross-dataset comparison on a matched forget class.
"""
import glob
import os

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

VKITTI_ROOT = ("/home/kavyansh/MySSD/research project/russo_work/machine_unlearning/"
               "project3/data/VKITTI2")

IGNORE = 255
_IMNET_MEAN = [0.485, 0.456, 0.406]
_IMNET_STD = [0.229, 0.224, 0.225]

# official VKITTI2 15-class palette (RGB -> class name), verified against
# sampled classSegmentation PNGs from this download.
COLOR_TO_NAME = {
    (0, 0, 0): "Undefined",
    (210, 0, 200): "Terrain",
    (90, 200, 255): "Sky",
    (0, 199, 0): "Tree",
    (90, 240, 0): "Vegetation",
    (140, 140, 140): "Building",
    (100, 60, 100): "Road",
    (250, 100, 255): "GuardRail",
    (255, 255, 0): "TrafficSign",
    (200, 200, 0): "TrafficLight",
    (255, 130, 0): "Pole",
    (80, 80, 80): "Misc",
    (160, 60, 60): "Truck",
    (255, 127, 80): "Car",
    (0, 139, 139): "Van",
}
NAMES = sorted(n for n in COLOR_TO_NAME.values() if n != "Undefined")
NAME_TO_IDX = {n: i for i, n in enumerate(NAMES)}
NUM_CLASSES = len(NAMES)  # 14

CAR = NAME_TO_IDX["Car"]
CARCLS = int(os.environ.get("VK_FORGET_CLS", str(CAR)))  # env-overridable, mirrors PERSON

SCENES = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
VARIANTS = ["clone", "fog", "morning", "overcast", "rain", "sunset",
            "15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right"]
# held-out scene for val, mirroring VSPW's video-level (not frame-level) split
TRAIN_SCENES = ["Scene01", "Scene02", "Scene06", "Scene18"]
VAL_SCENES = ["Scene20"]

# packed-RGB (r<<16 | g<<8 | b) -> class index, for a fast vectorized decode
_PACKED_LUT = {}
for _color, _name in COLOR_TO_NAME.items():
    _r, _g, _b = _color
    _PACKED_LUT[(_r << 16) | (_g << 8) | _b] = IGNORE if _name == "Undefined" else NAME_TO_IDX[_name]


def decode_label(png_path):
    """RGB classSegmentation PNG -> [H,W] int64 class-index array."""
    arr = np.array(Image.open(png_path).convert("RGB")).astype(np.int64)
    packed = (arr[..., 0] << 16) | (arr[..., 1] << 8) | arr[..., 2]
    out = np.full(packed.shape, IGNORE, dtype=np.int64)
    for code, idx in _PACKED_LUT.items():
        out[packed == code] = idx
    return out


def merge_car_label(label, car_idx=None, ignore=IGNORE):
    """Dominant-neighbor relabeling, identical algorithm to VSPW's
    merge_person_label, parameterized by the forget class index."""
    if car_idx is None:
        car_idx = CARCLS
    mask = label == car_idx
    if not mask.any():
        return label
    valid = (label != car_idx) & (label != ignore)
    if not valid.any():
        return label
    inds = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = label[tuple(inds)]
    out = label.copy()
    out[mask] = filled[mask]
    return out


def list_sequences(scenes=None, root=VKITTI_ROOT):
    """One 'video' = one (scene, variant, camera) tuple. Returns
    [(video_id, rgb_dir, seg_dir), ...]."""
    scenes = scenes or SCENES
    out = []
    for sc in scenes:
        for var in VARIANTS:
            for cam in ("Camera_0", "Camera_1"):
                rgb_dir = os.path.join(root, sc, var, "frames", "rgb", cam)
                seg_dir = os.path.join(root, sc, var, "frames", "classSegmentation", cam)
                if os.path.isdir(rgb_dir) and os.path.isdir(seg_dir):
                    out.append((f"{sc}_{var}_{cam}", rgb_dir, seg_dir))
    return out


def _pairs_for(rgb_dir, seg_dir):
    """rgb_%05d.jpg <-> classgt_%05d.png, matched by numeric suffix."""
    rgbs = sorted(glob.glob(os.path.join(rgb_dir, "rgb_*.jpg")))
    pairs = []
    for rp in rgbs:
        idx = os.path.splitext(os.path.basename(rp))[0].split("_")[1]
        sp = os.path.join(seg_dir, f"classgt_{idx}.png")
        if os.path.exists(sp):
            pairs.append((rp, sp))
    return pairs


class VKittiSeg(Dataset):
    """Per-frame dataset: (image[3,H,W], label[H,W]) i.i.d. over all frames of
    all sequences in the split. Direct analog of VSPWSeg."""

    def __init__(self, split="train", target="orig", size=512, crop=True,
                 car_only=False, root=VKITTI_ROOT, cache=True):
        scenes = TRAIN_SCENES if split == "train" else VAL_SCENES
        self.frames = []
        for vid, rgb_dir, seg_dir in list_sequences(scenes, root):
            self.frames.extend(_pairs_for(rgb_dir, seg_dir))
        self.split = split
        self.target = target
        self.size = size
        self.crop = crop and split == "train"
        if car_only:
            self.frames = self._filter_car_cached(root, split, cache)

    def _filter_car_cached(self, root, split, cache):
        import pickle
        cache_dir = os.path.join(root, ".clip_cache")
        cpath = os.path.join(cache_dir, f"{split}_frames_cls{CARCLS}.pkl")
        if cache and os.path.exists(cpath):
            rel = pickle.load(open(cpath, "rb"))
            keep = {(os.path.join(root, i), os.path.join(root, m)) for i, m in rel}
            return [fp for fp in self.frames if fp in keep]
        kept = [fp for fp in self.frames if self._has_car(fp[1])]
        if cache:
            os.makedirs(cache_dir, exist_ok=True)
            rel = [(os.path.relpath(i, root), os.path.relpath(m, root)) for i, m in kept]
            pickle.dump(rel, open(cpath, "wb"))
        return kept

    @staticmethod
    def _has_car(ann_path):
        return (decode_label(ann_path) == CARCLS).any()

    def __len__(self):
        return len(self.frames)

    def _load(self, idx):
        img_path, ann_path = self.frames[idx]
        img = Image.open(img_path).convert("RGB")
        label = decode_label(ann_path)
        return img, label

    def _sample_geom(self):
        if not self.crop:
            return None
        s = self.size
        scale = 0.5 + 1.5 * torch.rand(1).item()
        new = max(int(round(s * scale)), s)
        top = int(torch.randint(0, new - s + 1, (1,)).item())
        left = int(torch.randint(0, new - s + 1, (1,)).item())
        do_flip = torch.rand(1).item() < 0.5
        return (new, top, left, do_flip)

    def _apply(self, img, label_pil, geom):
        s = self.size
        if geom is not None:
            new, top, left, do_flip = geom
            img = TF.resize(img, [new, new], TF.InterpolationMode.BILINEAR)
            label_pil = TF.resize(label_pil, [new, new], TF.InterpolationMode.NEAREST)
            img = TF.crop(img, top, left, s, s)
            label_pil = TF.crop(label_pil, top, left, s, s)
            if do_flip:
                img = TF.hflip(img); label_pil = TF.hflip(label_pil)
        else:
            img = TF.resize(img, [s, s], TF.InterpolationMode.BILINEAR)
            label_pil = TF.resize(label_pil, [s, s], TF.InterpolationMode.NEAREST)
        img_t = TF.normalize(TF.to_tensor(img), _IMNET_MEAN, _IMNET_STD)
        label_t = torch.from_numpy(np.array(label_pil)).long()
        return img_t, label_t

    def __getitem__(self, idx):
        img, label = self._load(idx)
        if self.target == "merged":
            label = merge_car_label(label)
        label_pil = Image.fromarray(label.astype(np.int32), mode="I")
        geom = self._sample_geom()
        return self._apply(img, label_pil, geom)


class VKittiUnlearn(Dataset):
    """(image, orig_label, merged_label) over car-containing frames. Direct
    analog of VSPWUnlearn, with the same crop-after-merge ordering fix."""

    def __init__(self, split="train", size=512, crop=True, root=VKITTI_ROOT):
        self.base = VKittiSeg(split, target="orig", size=size, crop=crop,
                              car_only=True, root=root)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base._load(idx)
        merged_np = merge_car_label(label.copy())
        orig_pil = Image.fromarray(label.astype(np.int32), mode="I")
        merged_pil = Image.fromarray(merged_np.astype(np.int32), mode="I")
        geom = self.base._sample_geom()
        img_t, orig_t = self.base._apply(img, orig_pil, geom)
        _, merged_t = self.base._apply(img, merged_pil, geom)
        return img_t, orig_t, merged_t


def _build_clips(seqs, clip_len, stride, car_only):
    clips = []
    for vid, rgb_dir, seg_dir in seqs:
        pairs = _pairs_for(rgb_dir, seg_dir)
        for start in range(0, len(pairs) - clip_len + 1, stride):
            clip = pairs[start:start + clip_len]
            if car_only and not any(VKittiSeg._has_car(mp) for _, mp in clip):
                continue
            clips.append(clip)
    return clips


class VKittiClip(Dataset):
    """Clip of `clip_len` consecutive frames from one sequence, clip-consistent
    augmentation. Direct analog of VSPWClip."""

    def __init__(self, split="train", size=512, clip_len=3, stride=None,
                 crop=True, car_only=True, root=VKITTI_ROOT, cache=True):
        stride = clip_len if stride is None else stride
        self.clips = self._load_or_build(root, split, clip_len, stride, car_only, cache)
        self.clip_len = clip_len
        self.size = size
        self.crop = crop and split == "train"

    @staticmethod
    def _load_or_build(root, split, clip_len, stride, car_only, cache):
        import pickle
        cache_dir = os.path.join(root, ".clip_cache")
        key = f"{split}_len{clip_len}_str{stride}_car{int(car_only)}_cls{CARCLS}.pkl"
        cpath = os.path.join(cache_dir, key)
        if cache and os.path.exists(cpath):
            rel = pickle.load(open(cpath, "rb"))
            return [[(os.path.join(root, i), os.path.join(root, m)) for i, m in clip]
                    for clip in rel]
        scenes = TRAIN_SCENES if split == "train" else VAL_SCENES
        seqs = list_sequences(scenes, root)
        clips = _build_clips(seqs, clip_len, stride, car_only)
        if cache:
            os.makedirs(cache_dir, exist_ok=True)
            rel = [[(os.path.relpath(i, root), os.path.relpath(m, root)) for i, m in clip]
                   for clip in clips]
            pickle.dump(rel, open(cpath, "wb"))
        return clips

    def __len__(self):
        return len(self.clips)

    def _sample_geom(self):
        if not self.crop:
            return None
        s = self.size
        scale = 0.5 + 1.5 * torch.rand(1).item()
        new = max(int(round(s * scale)), s)
        top = int(torch.randint(0, new - s + 1, (1,)).item())
        left = int(torch.randint(0, new - s + 1, (1,)).item())
        do_flip = torch.rand(1).item() < 0.5
        return (new, top, left, do_flip)

    def _apply(self, img, label, geom):
        s = self.size
        if geom is not None:
            new, top, left, do_flip = geom
            img = TF.resize(img, [new, new], TF.InterpolationMode.BILINEAR)
            label = TF.resize(label, [new, new], TF.InterpolationMode.NEAREST)
            img = TF.crop(img, top, left, s, s); label = TF.crop(label, top, left, s, s)
            if do_flip:
                img = TF.hflip(img); label = TF.hflip(label)
        else:
            img = TF.resize(img, [s, s], TF.InterpolationMode.BILINEAR)
            label = TF.resize(label, [s, s], TF.InterpolationMode.NEAREST)
        img = TF.normalize(TF.to_tensor(img), _IMNET_MEAN, _IMNET_STD)
        label = torch.from_numpy(np.array(label)).long()
        return img, label

    def __getitem__(self, idx):
        geom = self._sample_geom()
        imgs, origs, mergeds = [], [], []
        for img_path, ann_path in self.clips[idx]:
            img = Image.open(img_path).convert("RGB")
            lab = decode_label(ann_path)
            merged_np = merge_car_label(lab.copy())
            lab_pil = Image.fromarray(lab.astype(np.int32), mode="I")
            merged_pil = Image.fromarray(merged_np.astype(np.int32), mode="I")
            img_t, orig_t = self._apply(img, lab_pil, geom)
            _, merged_t = self._apply(img, merged_pil, geom)
            imgs.append(img_t); origs.append(orig_t); mergeds.append(merged_t)
        return (torch.stack(imgs), torch.stack(origs), torch.stack(mergeds))


def _load_or_build_anchors(root, split, min_video_len=16, cache=True):
    """Anchor = one (video, frame_index) whose frame contains CARCLS. Direct
    analog of VSPW's _load_or_build_anchors."""
    import pickle
    cache_dir = os.path.join(root, ".clip_cache")
    cpath = os.path.join(cache_dir, f"{split}_anchors_cls{CARCLS}_minlen{min_video_len}.pkl")
    if cache and os.path.exists(cpath):
        vids_rel, anchors = pickle.load(open(cpath, "rb"))
        vids = {v: [(os.path.join(root, i), os.path.join(root, m)) for i, m in p]
                for v, p in vids_rel.items()}
        return vids, anchors

    pf_path = os.path.join(cache_dir, f"{split}_frames_cls{CARCLS}.pkl")
    car = None
    if os.path.exists(pf_path):
        car = {m for _i, m in pickle.load(open(pf_path, "rb"))}

    scenes = TRAIN_SCENES if split == "train" else VAL_SCENES
    vids, anchors = {}, []
    for vid, rgb_dir, seg_dir in list_sequences(scenes, root):
        pairs = _pairs_for(rgb_dir, seg_dir)
        if len(pairs) < min_video_len:
            continue
        hits = []
        for k, (_ip, ap) in enumerate(pairs):
            ok = (os.path.relpath(ap, root) in car) if car is not None \
                else VKittiSeg._has_car(ap)
            if ok:
                hits.append(k)
        if not hits:
            continue
        vids[vid] = pairs
        anchors.extend((vid, k) for k in hits)

    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        vids_rel = {v: [(os.path.relpath(i, root), os.path.relpath(m, root))
                        for i, m in p] for v, p in vids.items()}
        pickle.dump((vids_rel, anchors), open(cpath, "wb"))
    return vids, anchors


class VKittiClipAnchored(VKittiClip):
    """Clips of length T centred on a fixed anchor set. Direct analog of
    VSPWClipAnchored -- see that class's docstring for why this exists
    (evaluation-set composition must not vary with clip length)."""

    def __init__(self, split="train", size=512, clip_len=3, crop=False,
                 root=VKITTI_ROOT, cache=True, max_anchors=None, seed=0,
                 min_video_len=16, frame_stride=1):
        span = (clip_len - 1) * frame_stride + 1
        if span > min_video_len:
            raise ValueError(
                f"clip_len={clip_len} x frame_stride={frame_stride} spans {span} "
                f"frames but min_video_len={min_video_len}; raise min_video_len to "
                f">={span}, using that same value for every arm of the comparison.")
        videos, anchors = _load_or_build_anchors(root, split, min_video_len, cache)
        if max_anchors is not None and len(anchors) > max_anchors:
            g = torch.Generator().manual_seed(seed)
            keep = torch.randperm(len(anchors), generator=g)[:max_anchors].tolist()
            anchors = [anchors[i] for i in sorted(keep)]
        half = (span - 1) // 2
        self.clips = []
        for vid, k in anchors:
            pairs = videos[vid]
            start = max(0, min(k - half, len(pairs) - span))
            self.clips.append([pairs[start + i * frame_stride] for i in range(clip_len)])
        self.anchors = anchors
        self.clip_len = clip_len
        self.frame_stride = frame_stride
        self.size = size
        self.crop = crop and split == "train"


def verify_layout(root=VKITTI_ROOT, n_check=3):
    problems = []
    for split, scenes in (("train", TRAIN_SCENES), ("val", VAL_SCENES)):
        seqs = list_sequences(scenes, root)
        if not seqs:
            problems.append(f"{split}: zero sequences found for scenes {scenes}")
            continue
        for vid, rgb_dir, seg_dir in seqs[:n_check]:
            pairs = _pairs_for(rgb_dir, seg_dir)
            if not pairs:
                problems.append(f"{split}: sequence {vid!r} has 0 matched frame pairs")
        print(f"[verify_layout] {split}: {len(seqs)} sequences "
              f"(scenes={scenes}), checked first {min(n_check, len(seqs))}")
    if problems:
        print("[verify_layout] PROBLEMS FOUND:")
        for p in problems:
            print(f"  - {p}")
        return False
    print("[verify_layout] OK")
    return True


if __name__ == "__main__":
    verify_layout()
