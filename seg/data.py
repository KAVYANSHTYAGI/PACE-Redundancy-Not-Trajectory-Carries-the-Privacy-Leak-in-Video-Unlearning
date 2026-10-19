"""VSPW video semantic-segmentation data for class-unlearning (video analog of
project1/seg/data.py).

Conventions (mirrors project1's ADE20KSeg exactly, verified against the official
VSPW mmseg dataset class used by CFFM/TV3S):
  * 124 valid classes, label index 0..123 (raw PNG value minus 1).
  * raw value 0 -> "others"/unlabeled -> IGNORE (255); this is VSPW's
    `reduce_zero_label=True` convention, identical in spirit to ADE20K's.
  * PERSON is raw class 61 -> index 60 (confirmed against the official CLASSES
    tuple: index 60 in the 124-class list with "others" removed).

Directory layout (the full-resolution release -- the 480p release used by
CFFM/TV3S/MRCFA was Google-Drive-quota-blocked at download time; full-res is
functionally equivalent since every frame is resized to `size` in the loader
anyway, just slower to decode per-JPEG. Confirmed identical directory layout):
  VSPW/
    data/
      <video_id>/
        origin/*.jpg   (frames)
        mask/*.png     (per-frame semantic labels, same stem as origin)
    train.txt / val.txt / test.txt   (one video_id per line)

The unlearning target is built by the identical **dominant-neighbor relabeling**
used in project1 (eq. 1 in the paper), applied independently per frame: every
person pixel is replaced by the label of its nearest non-person, non-ignore
pixel *within that frame*. Whether cross-frame (temporal) relabeling is needed
is exactly the open question the temporal-attacker experiments are designed to
answer (see POSITIONING.md) -- this per-frame oracle is the correct starting
point, analogous to project1's per-image oracle, not a shortcut.

CONFIRMED 2026-07-21 against the real extracted archive (see verify_layout()
output logged this session): train.txt/val.txt/test.txt are exactly one
video_id per line (e.g. "0_wHveSGjXyDY"), 2806/343/387 videos respectively
(3536 total, matches official count). person raw pixel value = 61 (label_num_
dic_final.json), others = 0 -> confirms PERSON=60, NUM_CLASSES=124 after the
-1/ignore shift below. train/val videos have full origin/mask parity; test
videos ship origin frames with little/no mask (official held-out test split --
expected, we only use train/val).
"""
import glob
import os

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

VSPW_ROOT = ("/home/kavyansh/MySSD/research project/russo_work/machine_unlearning/"
             "project3/data/VSPW")

import os as _os
PERSON = int(_os.environ.get("FORGET_CLS", "60"))
IGNORE = 255
NUM_CLASSES = int(_os.environ.get("NUM_CLASSES", "124"))

_IMNET_MEAN = [0.485, 0.456, 0.406]
_IMNET_STD = [0.229, 0.224, 0.225]


def merge_person_label(label, person_idx=None, ignore=IGNORE):
    """Identical to project1's merge_person_label: nearest-valid-neighbor fill,
    per-frame. Kept as a free function (not tied to video structure) so it can
    be swapped for a temporal variant later without touching the dataset class."""
    if person_idx is None:
        person_idx = PERSON
    mask = label == person_idx
    if not mask.any():
        return label
    valid = (label != person_idx) & (label != ignore)
    if not valid.any():
        return label
    inds = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = label[tuple(inds)]
    out = label.copy()
    out[mask] = filled[mask]
    return out


def _read_split(split_path):
    """One video_id per line; tolerant of trailing whitespace/commas/extra
    columns. See module docstring's NOTE on split-file format."""
    ids = []
    with open(split_path) as f:
        for line in f:
            tok = line.strip().split(",")[0].split()[0] if line.strip() else ""
            if tok:
                ids.append(tok)
    return ids


class VSPWSeg(Dataset):
    """Per-frame dataset: returns (image[3,H,W] float, label[H,W] long), i.i.d.
    over all frames of all videos in the split. This is the B0 baseline dataset
    -- structurally identical to project1's ADE20KSeg, just aggregated across
    videos instead of a flat image folder. Clip-level (temporal) sampling is a
    separate dataset class added once this per-frame floor is established
    (POSITIONING.md step 3-4), not a flag on this one.

    target="orig"   -> original labels (train the original model)
    target="merged" -> person dominant-neighbor relabeled, per-frame (oracle)
    person_only=True -> restrict to frames that contain a person
    """

    def __init__(self, split="train", target="orig", size=512, crop=True,
                 person_only=False, root=VSPW_ROOT, cache=True):
        split_file = os.path.join(root, f"{split}.txt")
        video_ids = _read_split(split_file)
        self.frames = []  # list of (img_path, mask_path)
        n_orphan = 0
        for vid in video_ids:
            img_dir = os.path.join(root, "data", vid, "origin")
            ann_dir = os.path.join(root, "data", vid, "mask")
            for ann_path in sorted(glob.glob(os.path.join(ann_dir, "*.png"))):
                stem = os.path.splitext(os.path.basename(ann_path))[0]
                img_path = os.path.join(img_dir, stem + ".jpg")
                # 2/2806 train videos ship one orphan mask with no matching jpg
                # (known corrupted-frame issue, see VSPW-dataset-download repo's
                # supplementary fix file) -- skip rather than crash mid-training.
                if os.path.exists(img_path):
                    self.frames.append((img_path, ann_path))
                else:
                    n_orphan += 1
        if n_orphan:
            print(f"[VSPWSeg] skipped {n_orphan} orphan mask(s) with no matching frame")
        self.split = split
        self.target = target
        self.size = size
        self.crop = crop and split == "train"
        if person_only:
            self.frames = self._filter_person_cached(root, split, cache)

    def _filter_person_cached(self, root, split, cache):
        """The person_only scan is a single-threaded pass over every mask in the
        split (~198k files for train, several minutes) -- every script in
        run_all.sh (baselines, ADVR, ADVR-T, benchmark, temporal-attack) builds
        this independently, so cache it (mirrors VSPWClip's clip-index cache).

        Cache key includes PERSON (the forget-class index) -- this used to be
        the literal string f"{split}_person_frames.pkl" regardless of class,
        so switching FORGET_CLS (e.g. to test generalization to a class other
        than person) would either silently reuse the wrong class's frame list
        or overwrite the existing person-class cache. Keyed by class now so
        different forget classes can never collide."""
        import pickle
        cache_dir = os.path.join(root, ".clip_cache")
        cpath = os.path.join(cache_dir, f"{split}_frames_cls{PERSON}.pkl")
        if cache and os.path.exists(cpath):
            rel = pickle.load(open(cpath, "rb"))
            keep = {(os.path.join(root, i), os.path.join(root, m)) for i, m in rel}
            return [fp for fp in self.frames if fp in keep]
        kept = [fp for fp in self.frames if self._has_person(fp[1])]
        if cache:
            os.makedirs(cache_dir, exist_ok=True)
            rel = [(os.path.relpath(i, root), os.path.relpath(m, root)) for i, m in kept]
            pickle.dump(rel, open(cpath, "wb"))
        return kept

    @staticmethod
    def _has_person(ann_path):
        return (np.array(Image.open(ann_path)) == PERSON + 1).any()

    def __len__(self):
        return len(self.frames)

    def _load(self, idx):
        img_path, ann_path = self.frames[idx]
        img = Image.open(img_path).convert("RGB")
        raw = np.array(Image.open(ann_path)).astype(np.int64)
        label = raw - 1
        # VSPW uses raw=0 ("others") and raw=255 (explicit void) as ignore, AND
        # a full-dataset scan (2026-07-21) found 17 val masks with stray raw=253
        # (undocumented, not one of the 124 classes -- real annotation corruption,
        # not a code bug: e.g. data/353_-5sqmCF8OFU/mask/00000404.png). A small
        # sample missed this; only a full scan caught it. Any raw value outside
        # {0} u {1..124} u {255} is therefore treated as ignore, not just 0/255,
        # so future stray corrupted IDs fail safe instead of crashing training.
        label[(raw == 0) | (raw > 124)] = IGNORE
        return img, label

    def _sample_geom(self):
        """One geometry: (scale, top, left, do_flip), or None for eval (no crop).
        Factored out (mirrors VSPWClip._sample_geom/_apply) so VSPWUnlearn can
        share ONE crop across img/orig/merged -- see _apply's docstring for why
        that sharing has to happen before merge_person_label, not after."""
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
        """Apply `geom` (or plain resize if geom is None) to one image + one
        label. Caller must run merge_person_label on the FULL-RESOLUTION label
        BEFORE calling this, never after: cropping first and merging second
        (the bug this replaces) can crop out every non-person, non-ignore
        pixel, at which point merge_person_label's nearest-neighbour fill has
        nothing valid to grab and silently returns the label UNCHANGED -- the
        "unlearning target" for that sample then still says PERSON. Verified:
        an all-person 64x64 crop reproduces this exactly (merged == orig).
        Because the augmentation here can zoom in up to 2x before cropping,
        an all-person (or all-person-plus-ignore) crop is not a corner case --
        anywhere a person fills a large fraction of the frame, zooming in
        further makes it close to certain."""
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
            label = merge_person_label(label)          # BEFORE crop -- see _apply
        label_pil = Image.fromarray(label.astype(np.int32), mode="I")
        geom = self._sample_geom()
        return self._apply(img, label_pil, geom)


class VSPWUnlearn(Dataset):
    """(image, orig_label, merged_label) over person-containing frames, for the
    unlearning methods -- direct analog of project1's ADE20KUnlearn.

    BUG FIXED 2026-07-30: this used to build `orig` via self.base[idx] (which
    applies VSPWSeg's random crop) and only THEN run merge_person_label on the
    already-cropped label. That silently corrupts the unlearning target: if the
    crop (this augmentation can zoom in up to 2x before cropping, so this is
    common whenever a person fills a large part of the frame) contains no
    valid non-person/non-ignore pixel, merge_person_label's nearest-neighbour
    fill has nothing to grab and returns the label UNCHANGED -- the "forget"
    target then still says PERSON for that sample, training every method that
    consumes it (all 8 baselines, plain ADVR) to KEEP predicting the class on
    exactly the crops where forgetting matters most. VSPWClip (used by
    ADVR-T/ADVR-TX) never had this bug -- it already computed the merge before
    cropping. Fixed here to do the same: merge on the full-resolution label,
    then apply ONE sampled crop geometry to img/orig/merged together.
    """

    def __init__(self, split="train", size=512, crop=True, root=VSPW_ROOT):
        self.base = VSPWSeg(split, target="orig", size=size, crop=crop,
                            person_only=True, root=root)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base._load(idx)                      # full-res, ignore-fixed
        merged_np = merge_person_label(label.copy())            # BEFORE crop
        orig_pil = Image.fromarray(label.astype(np.int32), mode="I")
        merged_pil = Image.fromarray(merged_np.astype(np.int32), mode="I")
        geom = self.base._sample_geom()                         # ONE geometry
        img_t, orig_t = self.base._apply(img, orig_pil, geom)
        _, merged_t = self.base._apply(img, merged_pil, geom)   # same geom
        return img_t, orig_t, merged_t


# --------------------------------------------------------------------------- #
# Temporal / clip datasets -- the video-specific part with no project1 analog.
# A "clip" is `clip_len` CONSECUTIVE frames from ONE video. The temporal
# attacker (seg/attack.py TemporalSilhouetteNet) and ADVR-T (seg/candidates.py)
# consume these to probe / defend trajectory-level leakage, per POSITIONING.md's
# temporal HYP: I(M_{1:T}; O_{1:T}) can be > 0 even when every per-frame
# I(M_t; O_t) -> 0. Augmentation is applied IDENTICALLY across all frames of a
# clip (one sampled scale/crop/flip per clip) so temporal correspondence is
# preserved -- a per-frame-random transform would destroy the very trajectory
# signal these datasets exist to measure.
# --------------------------------------------------------------------------- #
def _build_clips(video_ids, root, clip_len, stride, person_only):
    """Return a list of clips, each a list of `clip_len` (img_path, mask_path)
    tuples of consecutive frames from one video. `stride` is the gap between clip
    start indices (stride>=clip_len -> non-overlapping)."""
    clips = []
    for vid in video_ids:
        img_dir = os.path.join(root, "data", vid, "origin")
        ann_dir = os.path.join(root, "data", vid, "mask")
        anns = sorted(glob.glob(os.path.join(ann_dir, "*.png")))
        pairs = []
        for ann_path in anns:
            stem = os.path.splitext(os.path.basename(ann_path))[0]
            img_path = os.path.join(img_dir, stem + ".jpg")
            if os.path.exists(img_path):
                pairs.append((img_path, ann_path))
        for start in range(0, len(pairs) - clip_len + 1, stride):
            clip = pairs[start:start + clip_len]
            if person_only:
                # keep the clip iff person appears in >=1 frame (trajectory needs
                # at least one anchor; the attacker predicts the mask at every
                # frame including empty ones, which is the point).
                if not any(VSPWSeg._has_person(mp) for _, mp in clip):
                    continue
            clips.append(clip)
    return clips


class VSPWClip(Dataset):
    """Returns a clip: (imgs[T,3,H,W], origs[T,H,W], mergeds[T,H,W]) of `clip_len`
    consecutive frames from one video, with clip-consistent augmentation.

    Used both for the temporal-attacker protocol (which reads the model's
    per-frame output field stacked over the clip) and for ADVR-T training. Set
    target-independent: it always returns both orig and merged (per-frame
    dominant-neighbour relabel), like VSPWUnlearn, since the temporal apparatus
    needs both. person_only restricts to clips containing >=1 person frame.
    """

    def __init__(self, split="train", size=512, clip_len=3, stride=None,
                 crop=True, person_only=True, root=VSPW_ROOT, cache=True):
        stride = clip_len if stride is None else stride
        self.clips = self._load_or_build(root, split, clip_len, stride,
                                         person_only, cache)
        self.clip_len = clip_len
        self.size = size
        self.crop = crop and split == "train"

    @staticmethod
    def _load_or_build(root, split, clip_len, stride, person_only, cache):
        """Clip-index build scans every mask file (~7min for train); cache it so
        repeated temporal experiments don't re-pay that. Cache key covers every
        parameter that changes the index; stored as paths RELATIVE to root so the
        cache survives a dataset move.

        Includes PERSON (the forget-class index) in the key -- when
        person_only=True the index depends on _build_clips' class-filtering,
        which depends on the current FORGET_CLS. Without this, switching
        FORGET_CLS (e.g. to test generalization to a class other than person)
        would silently reuse a stale, wrong-class clip index."""
        import pickle
        cache_dir = os.path.join(root, ".clip_cache")
        key = f"{split}_len{clip_len}_str{stride}_person{int(person_only)}_cls{PERSON}.pkl"
        cpath = os.path.join(cache_dir, key)
        if cache and os.path.exists(cpath):
            rel = pickle.load(open(cpath, "rb"))
            return [[(os.path.join(root, i), os.path.join(root, m)) for i, m in clip]
                    for clip in rel]
        video_ids = _read_split(os.path.join(root, f"{split}.txt"))
        clips = _build_clips(video_ids, root, clip_len, stride, person_only)
        if cache:
            os.makedirs(cache_dir, exist_ok=True)
            rel = [[(os.path.relpath(i, root), os.path.relpath(m, root)) for i, m in clip]
                   for clip in clips]
            pickle.dump(rel, open(cpath, "wb"))
        return clips

    def __len__(self):
        return len(self.clips)

    def _sample_geom(self):
        """One geometry per clip: (scale, top, left, do_flip) or None for eval."""
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
            raw = np.array(Image.open(ann_path)).astype(np.int64)
            lab = raw - 1
            lab[(raw == 0) | (raw > 124)] = IGNORE  # see VSPWSeg._load NOTE
            merged_np = merge_person_label(lab.copy())
            lab_pil = Image.fromarray(lab.astype(np.int32), mode="I")
            merged_pil = Image.fromarray(merged_np.astype(np.int32), mode="I")
            img_t, orig_t = self._apply(img, lab_pil, geom)
            _, merged_t = self._apply(img, merged_pil, geom)  # same geom
            imgs.append(img_t); origs.append(orig_t); mergeds.append(merged_t)
        return (torch.stack(imgs), torch.stack(origs), torch.stack(mergeds))


def _load_or_build_anchors(root, split, min_video_len=16, cache=True):
    """Anchor = one (video, frame_index) whose frame contains the forget class
    (PERSON -- despite the name, this is whatever FORGET_CLS currently is).

    The anchor set depends ONLY on the split (and forget class), never on clip
    length -- that is the whole point (see VSPWClipAnchored). Reuses the
    existing per-class frame-filter cache so this costs a directory walk, not
    a re-scan of every mask.

    Cache key includes PERSON: this used to be f"{split}_anchors_minlen{L}.pkl"
    with no class in it, so switching FORGET_CLS (e.g. to test generalization
    to a class other than person) would either silently reuse the wrong
    class's anchors or overwrite the existing person-class cache. Keyed by
    class now so different forget classes can never collide.
    """
    import pickle
    cache_dir = os.path.join(root, ".clip_cache")
    cpath = os.path.join(cache_dir, f"{split}_anchors_cls{PERSON}_minlen{min_video_len}.pkl")
    if cache and os.path.exists(cpath):
        vids_rel, anchors = pickle.load(open(cpath, "rb"))
        vids = {v: [(os.path.join(root, i), os.path.join(root, m)) for i, m in p]
                for v, p in vids_rel.items()}
        return vids, anchors

    pf_path = os.path.join(cache_dir, f"{split}_frames_cls{PERSON}.pkl")
    person = None
    if os.path.exists(pf_path):
        person = {m for _i, m in pickle.load(open(pf_path, "rb"))}

    vids, anchors = {}, []
    for vid in _read_split(os.path.join(root, f"{split}.txt")):
        img_dir = os.path.join(root, "data", vid, "origin")
        ann_dir = os.path.join(root, "data", vid, "mask")
        pairs = []
        for ann_path in sorted(glob.glob(os.path.join(ann_dir, "*.png"))):
            stem = os.path.splitext(os.path.basename(ann_path))[0]
            img_path = os.path.join(img_dir, stem + ".jpg")
            if os.path.exists(img_path):
                pairs.append((img_path, ann_path))
        # require enough frames to cut the LONGEST window we will ever test, so
        # the anchor set stays literally identical across the clip-length sweep
        if len(pairs) < min_video_len:
            continue
        hits = []
        for k, (_ip, ap) in enumerate(pairs):
            ok = (os.path.relpath(ap, root) in person) if person is not None \
                else VSPWSeg._has_person(ap)
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


class VSPWClipAnchored(VSPWClip):
    """Clips of length T CENTRED on a fixed anchor set, for clip-length sweeps.

    Why this replaces VSPWClip for the T-sweep. VSPWClip tiles each video with
    stride=clip_len, so the number of clips collapses as T grows (val: 8311 at
    T=2 down to 896 at T=16). Capping the loader at the first N clips then
    silently changes WHICH DATA each T is scored on -- measured on the real
    cache, the T=2 point came from 36 videos and the T=16 point from 238. A
    "leakage vs T" curve built that way is confounded with dataset composition,
    and the giveaway is that the retrain-from-scratch oracle floor -- a quantity
    with no dependence on T whatsoever -- drifted 17.7 -> 0.0 across the sweep.

    Here the anchors are drawn once, from the split alone, and every T extracts
    the window centred on the same anchors. Same videos, same frames, same
    person density at every T; only the window length changes. Windows are
    clamped inside the video, so an anchor near a boundary yields an off-centre
    but still valid window.
    """

    def __init__(self, split="train", size=512, clip_len=3, crop=False,
                 root=VSPW_ROOT, cache=True, max_anchors=None, seed=0,
                 min_video_len=16, frame_stride=1):
        """`frame_stride` samples every k-th frame instead of consecutive ones.

        Trajectory information scales with ELAPSED TIME, not with frame count:
        16 consecutive VSPW frames span ~1.1s, in which a walking person barely
        displaces, so there is almost no trajectory for a temporal attacker to
        exploit no matter how good it is. frame_stride=2 spans ~2.1s at the same
        compute and the same T. Any claim that "there is no long-range temporal
        leakage" is weak if every clip tested covered about one second.

        NOTE on keeping the sweep clean: a window needs (clip_len-1)*frame_stride+1
        frames, so raising either drops short videos and would silently change the
        anchor set -- exactly the confound §8.2 of POSITIONING.md is about. Pin
        min_video_len to the largest requirement across every configuration being
        compared (e.g. 31 for clip_len=16 at frame_stride=2) so all of them share
        one anchor set.
        """
        span = (clip_len - 1) * frame_stride + 1
        if span > min_video_len:
            raise ValueError(
                f"clip_len={clip_len} x frame_stride={frame_stride} spans {span} "
                f"frames but min_video_len={min_video_len}; raise min_video_len to "
                f">={span} (and use that same value for every arm of the comparison, "
                "or the anchor sets will differ)")
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


def verify_layout(root=VSPW_ROOT, n_check=5):
    """Sanity check to run once, immediately after extraction, before any
    training: confirms the split files parse to real video directories and
    that frame/mask pairs actually exist on disk. Catches a split-file format
    mismatch at setup time, not mid-training.

    `test` is VSPW's official held-out split and ships WITHOUT masks (confirmed
    2026-07-21: test videos have an origin/ dir but no mask/ dir at all) -- we
    only ever train/eval on train+val, so missing masks there are expected and
    not reported as a problem; only missing origin frames on test are checked."""
    problems = []
    for split in ("train", "val", "test"):
        split_file = os.path.join(root, f"{split}.txt")
        if not os.path.exists(split_file):
            problems.append(f"missing {split_file}")
            continue
        ids = _read_split(split_file)
        if not ids:
            problems.append(f"{split}.txt parsed to zero video ids")
            continue
        needs_mask = split != "test"
        for vid in ids[:n_check]:
            img_dir = os.path.join(root, "data", vid, "origin")
            ann_dir = os.path.join(root, "data", vid, "mask")
            if not os.path.isdir(img_dir):
                problems.append(f"{split}: video id {vid!r} has no origin dir")
                continue
            n_img = len(glob.glob(os.path.join(img_dir, "*.jpg")))
            if n_img == 0:
                problems.append(f"{split}: video {vid!r} origin dir has 0 frames")
            if needs_mask:
                if not os.path.isdir(ann_dir):
                    problems.append(f"{split}: video id {vid!r} has no mask dir")
                    continue
                n_ann = len(glob.glob(os.path.join(ann_dir, "*.png")))
                if n_ann == 0:
                    problems.append(f"{split}: video {vid!r} mask dir has 0 frames")
        print(f"[verify_layout] {split}: {len(ids)} videos listed, "
              f"checked first {min(n_check, len(ids))}"
              + ("" if needs_mask else " (no mask expected, official test split)"))
    if problems:
        print("[verify_layout] PROBLEMS FOUND:")
        for p in problems:
            print(f"  - {p}")
        return False
    print("[verify_layout] OK")
    return True


if __name__ == "__main__":
    verify_layout()
