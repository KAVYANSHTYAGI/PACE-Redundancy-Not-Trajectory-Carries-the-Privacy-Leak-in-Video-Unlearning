# Project3 Experiment Log: Temporal Silhouette Leakage in Video Unlearning

Living document. Every architecture, every protocol decision, every bug found,
every result, in enough detail to reproduce or to challenge. Ordered
chronologically within each section; most recent state is what to trust.
Companion to `POSITIONING.md` (research framing/direction), this file is the
experimental record; POSITIONING.md is the narrative.

---

## 1. The setting

**Task.** Video semantic segmentation on VSPW (124 classes, 3536 videos,
2806/343/387 train/val/test). Forget class: `person` (raw pixel 61 → index 60
after the standard `-1`/ignore shift). "Unlearning" = make a trained model stop
predicting `person` without retraining from scratch, and without damaging its
accuracy on everything else (the "retain" classes).

**Why video, not just images.** A companion project (project1) showed that
image segmentation models which appear to have "forgotten" a class under naive
inspection (label never predicted) still encode a recoverable *silhouette* of
where that class was, a small attacker network can reconstruct it from the
model's output field on the other classes. The question here: does an analogous
leak exist *across time* in video, i.e., can a model be safe on every single
frame and still leak the object's trajectory when an attacker is allowed to look
at several frames at once?

**Directory map.**
```
seg/data.py         VSPW datasets: per-frame (VSPWSeg), unlearning pairs
                    (VSPWUnlearn), clips (VSPWClip, VSPWClipAnchored)
seg/model.py        DeepLabV3-ResNet50 factory (get_seg_model, seg_forward)
seg/methods.py      Pixel-level unlearning baselines (FT, NegGrad, ..., SCRUB)
seg/baselines/      Same baselines, ported verbatim from project1
seg/candidates.py   ADVR, ADVR-T, ADVR-TX (the paper's method line)
seg/attack.py       All attacker architectures + train/eval loops
seg/mamba.py        Selective-state-space (Mamba) temporal mixing block
seg/metrics.py      Threshold-free scoring (AP/AUC/IoU_best), traj_consistency
seg/audit.py        leakage_metrics, relearn_audit, trajectory_leakage
scripts/            Drivers, see §7 for which ones are CURRENT vs SUPERSEDED
results/seg/        All JSON outputs; *_T{N}.json / temporal_attack*.json
                    from the SUPERSEDED protocol are kept on disk but must not
                    be cited (see §4)
```

---

## 2. Dataset and data pipeline

### 2.1 VSPW conventions
- 124 valid classes, raw PNG value − 1 = label index (`reduce_zero_label`).
- `PERSON = 60` (raw 61), confirmed against the official CLASSES tuple.
- Ignore = raw 0 ("others") **and** raw 255 (explicit void) **and** any raw
  value outside `{0} ∪ {1..124} ∪ {255}`, a full-split scan (2026-07-21) found
  17 val masks with a stray, undocumented raw value 253 (real annotation
  corruption). A 150-frame sample had missed this; only a full scan caught it.
  **Lesson:** never validate a new dataset's label space on a sample; scan the
  full split or budget for a mid-training crash.
- Test split ships frames with no masks (official held-out split, expected).

### 2.2 Dataset classes (`seg/data.py`)
- **`VSPWSeg`**: per-frame `(image, label)`, i.i.d. over all frames. Used for
  training the `original` and `oracle` reference models and for retain-replay.
- **`VSPWUnlearn`**: per-frame `(img, orig, merged)` where `merged` is the
  per-frame dominant-neighbor relabeling of `person` (nearest non-person,
  non-ignore label, via `scipy.ndimage.distance_transform_edt`). This relabeling
  defines the unlearning target for every method.
- **`VSPWClip`**: clips of `clip_len` consecutive frames, tiled per video with
  `stride` (default `stride=clip_len`, i.e. non-overlapping). Used for ADVR-T
  training. **Known limitation (see §4.1): not suitable for a clip-length sweep**
  because clip count collapses as `clip_len` grows.
- **`VSPWClipAnchored`** (added 2026-07-26, the fix for §4.1), clips of length
  `clip_len` **centred on a fixed anchor set** (one `(video, frame_index)` per
  anchor, drawn once from the split and cached), so every window length windows
  the *same* underlying content. Supports `frame_stride` (temporal dilation,
  see §4.3) and enforces `min_video_len ≥ (clip_len−1)·frame_stride + 1` so the
  anchor set never silently changes composition across a sweep.
  - Verified: anchors are byte-identical across `clip_len ∈ {2,3,5,8,16}` at
    fixed `max_anchors` (1200 clips, 231 videos, every time).
  - Verified: anchors are byte-identical across `frame_stride ∈ {1,2}` at fixed
    `min_video_len=31` (needed for `clip_len=16, stride=2`).

### 2.3 Caching layer
- `.clip_cache/{split}_person_frames.pkl`: person-frame filter (was rescanned
  from scratch, ~8min, by every pipeline stage; now cached).
- `.clip_cache/{split}_len{T}_str{S}_person{0,1}.pkl`: `VSPWClip` clip index.
- `.clip_cache/{split}_anchors_minlen{L}.pkl`: `VSPWClipAnchored` anchor set +
  per-video frame-pair lists.

---

## 3. Model and baselines

### 3.1 Reference models
- `original`: DeepLabV3-R50, trained normally, includes `person`.
- `oracle`: same architecture, retrained from scratch with `person` never a
  distinguishable class (labels relabeled before training starts). This is the
  **privacy floor**: the best any unlearning method could achieve, since even a
  model that never saw the class still encodes *some* prior about where a
  person-shaped hole tends to be (context correlations with other classes).

### 3.2 Published baselines (ported from project1, `seg/methods.py` /
`seg/baselines/`)
FT, NegGrad, NegGrad+, Random-Label (RL), SalUn (+ saliency masking), SSD,
SCRUB, DELETE. All operate on `VSPWUnlearn` triples; hyperparameters per method
in `scripts/run_baselines.py`'s `HP` dict. All 8 trained to completion on real
VSPW data (see §6.1 for wall-clock).

### 3.3 The method line
- **ADVR** (`seg/candidates.adv_replay`), ported from project1. Adversarial
  training against a **per-frame** silhouette discriminator (`SilhouetteNet`):
  blend loss (push person pixels to the dominant-neighbor label) + retain-KL
  anchor to a frozen teacher + adversarial term to fool the frame discriminator
  + retain-replay on ordinary frames. Makes each frame individually hard to
  attack.
- **ADVR-T** (`seg/candidates.adv_temporal`), identical, except the adversary
  is a **temporal** discriminator (`TemporalSilhouetteNet` or `ConvGRUAttacker`)
  that reconstructs the person-mask *trajectory* from a clip's field sequence,
  plus an optional temporal-consistency term `_temporal_consistency` (weight
  `lam_tc`): the forgotten region's frame-to-frame field change must not exceed
  the *mean* change over all retain pixels in the clip (a global reference).
- **ADVR-TX** (`seg/candidates.adv_temporal_x`, added 2026-07-26), two
  upgrades to ADVR-T's objective:
  1. **Local-ring `L_tc`** (`_temporal_consistency_local`), replaces the global
     mean reference with the mean change over a **dilated ring immediately
     surrounding the hole** (not the whole retain set). Rationale: real
     field-change is spatially heterogeneous (boundaries move more than flat
     interiors), so a global reference gives a hole in a busy region a lenient
     budget to hide motion in, while over-penalizing a hole in a static region
     for no reason.
  2. **`OrderDiscriminator`**: a second adversary that reads the field sequence
     masked to a dilated neighborhood of the hole and predicts **true order vs.
     time-permuted**. The model is trained to make that judgement impossible.
     Rationale (superseded, see §5.2): magnitude-based `L_tc` is a proxy; a
     model could respect a change budget and still move smoothly and
     *readably*. Exchangeability is the direct statement of "no order
     information," hence "no trajectory."
  - **Status (2026-07-27): not yet trained.** The Phase-A result in §5 shows the
    order channel this targets is empirically ~0, which means `OrderDiscriminator`
    is aimed at a mechanism that may not be the real leak. Do not train ADVR-TX
    as currently specified without first re-deriving its target, see §8, item 2.

---

## 4. Protocol defects found and fixed (2026-07-26 audit)

Triggered by a suspiciously clean "leakage strengthens monotonically with clip
length" curve. Caught by checking a quantity that is **required** to be
invariant to the swept variable (the retrain-oracle floor) and finding that it
moved. General lesson recorded in memory as
`invariant-quantities-catch-protocol-bugs`.

### 4.1 Defect 1, evaluation content varied with window length (severe)

`VSPWClip` tiles with `stride=clip_len`, so clip count collapses as `clip_len`
grows. The old protocol (`run_temporal_attack.py`) then capped the eval loader
at the first `N_VAL` clips (clips ordered by video). Measured on the real
cache:

| T | clips available | clips scored (cap 1500) | **distinct val videos** |
|---|---|---|---|
| 2 | 8311 | 1500 | **36** |
| 3 | 5557 | 1500 | 50 |
| 5 | 3335 | 1500 | 77 |
| 8 | 2008 | 1500 | 166 |
| 16 | 896 | 896 (all) | **238** |

T=2 was scored on 36 videos, T=16 on 238, a dataset-composition sweep, not a
window-length sweep. (Person-density per clip was checked as a possible second
composition confound and ruled out: 99.9% → 98.6% across T=2→16, not
meaningfully different.)

**Fix:** `VSPWClipAnchored` (§2.2).

### 4.2 Defect 2, attacker budget shrank with window length (severe)

`MAX_ITERS=500` was held fixed while `BS_CLIP` was manually scaled down for GPU
memory (8 → 4 → 2 → 1 across T=2→16, to avoid the OOM described in §6.2), so
clips actually seen by the attacker went 4000 → 2000 → 1000 → **500**, with
batch-size-1 gradients at T=16. A weaker-budget attacker finds less, so
measured leakage falls with T for any model, independent of the model's actual
properties.

**Fix:** `train_temporal_attacker(accum_steps=...)`, gradient accumulation
gives a fixed number of *optimizer steps* at a fixed *effective batch in clips*
regardless of the physical batch size forced by memory. Also logs
`D.train_losses` so a converged-but-empty attacker is distinguishable from one
that never trained (both would report ~0 leakage, for very different reasons).

### 4.3 Defect 3, IoU@0.5 is a calibration artifact, not a privacy measure

Under the old protocol, ADVR-T at T=16 scored **IoU 0.0**, read at the time as
"no leakage." Its AUC at the same point was **61.2**, i.e. above chance: the
attacker's *ranking* of person vs. non-person pixels was informative, its
sigmoid output simply never crossed the hardcoded 0.5 threshold. Reproduced in
a controlled synthetic case (`seg/metrics.py` module docstring / validation): an
attacker at AUC 94.7% scores IoU@0.5 = 2.3%, while its own best-threshold IoU is
37.7%. `traj_consistency` had the matching bug: `100 * (1 - flip/max(active,1))`
returned a perfect-looking `100.0` whenever the attacker predicted nothing
(`active=0`), this fired for the oracle floor at T=8 and for ADVR-T at T=16.

**Fix:** `seg/metrics.py`, `ScoreHist` accumulates an exact 512-bin score
histogram over **every** pixel (no subsampling) and derives AP (primary
metric), AUC, IoU@0.5 (kept for continuity), `IoU_best` (calibration-free
reconstruction quality, IoU at its own best threshold), `pos_rate` (fraction
predicted positive, diagnostic for all-negative collapse). `TrajConsistency`
returns NaN below a minimum-active-pixel threshold rather than a fake 100.
Validated against `sklearn.metrics.roc_auc_score` / `average_precision_score`
(exact AUC match, AP within histogram quantization) plus the synthetic
calibration case above.

### 4.4 Surviving confound, utility (NOT yet fixed, flagged for §8)

From `results/seg/benchmark.json`: `retain_mIoU` is oracle **30.8**, ADVR
**24.2**, ADVR-T **22.5**. ADVR-T's output field is globally degraded relative
to the oracle, part of its apparent "privacy" may be nothing more than there
being less information of *any kind* in its output for an attacker to read.
This is also consistent with ADVR-T scoring *below* a floor that is supposed to
be irreducible in some of the old (confounded) T-sweep numbers. **Leakage
comparisons are only meaningful at matched utility.** Not yet addressed
experimentally, see §8.3.

### 4.5 New control: separating redundancy from synergy

Even with 4.1–4.3 fixed, "temporal attacker beats per-frame attacker" is
ambiguous between two mechanisms:
1. **Redundancy / denoising**: the mask is temporally correlated, so averaging
   T noisy per-frame estimates cuts variance. Order-independent. Grows with T
   for *any* per-frame-leaky model. Not novel or interesting on its own.
2. **Synergy**: information carried by the trajectory itself (motion,
   differential cues) present in no single frame. Order-dependent. This is what
   "temporal silhouette leakage" was originally meant to claim.

**Control:** `_shuffle_time(O, M)` applies the *same* random time-permutation to
a clip's field sequence and its mask trajectory, independently per clip in the
batch (verified: identical permutation on O and M, valid permutation of each
row, independent across the batch, no-op at T=1). Training/evaluating on
shuffled clips destroys (2) while leaving (1) intact, so
`synergy = AP(ordered) − AP(shuffled)` isolates the genuinely trajectory-borne
component. This is the single most important addition to the protocol, see §5
for what it revealed.

---

## 5. Corrected-protocol results (current, trustworthy)

**Driver:** `scripts/run_temporal_protocol.py` (per-window-length run) +
`scripts/sweep_protocol.sh` (T sweep) + `scripts/run_attacker_zoo.sh` (attacker ×
span sweep). Config knobs: `CLIP_LEN`, `N_TRIALS`, `STEPS` (fixed optimizer
steps), `EFF_CLIPS` (fixed effective batch), `TEMPORAL_KIND`, `FRAME_STRIDE`,
`MIN_VIDEO_LEN`.

### 5.1 T=3, conv3d attacker, 0.2s span, `results/seg/protocol_T3.json`

5 models, `N_TRIALS=2`, `STEPS=500`, effective batch 4 clips.

| model | frame AP | shuffled AP | ordered AP | Δ_redundancy | **SYNERGY** |
|---|---|---|---|---|---|
| oracle_FLOOR | 24.73 | 26.77 | 26.99 | +2.04 | +0.22 |
| original | 91.56 | 94.94 | 94.94 | +3.38 | **+0.00** |
| ADVR | 29.44 | 45.35 | 45.26 | **+15.91** | −0.09 |
| ADVR-T | 16.42 | 22.62 | 21.26 | +6.20 | −1.36 |
| bl_SalUn_s1 | 49.27 | 83.54 | 84.74 | **+34.27** | +1.20 |

Additional per-arm detail available in the JSON: AUC, IoU@0.5, IoU_best,
pos_rate, `traj_consistency`, `final_train_loss` per trial-averaged arm.

**Reading:** synergy is at or below the floor's own synergy (+0.22) for every
model, no order effect detectable at T=3. Δ_redundancy is large and *scales
with how private the method looks per-frame*: SalUn (frame AP 49.3, near
`original`'s 91.6) jumps to 83.5–84.7 temporally (+34.3); ADVR (frame AP 29.4,
close to the floor's 24.7) jumps to 45.3–45.4 (+15.9), ending up **well above
the floor** despite looking nearly at-floor per-frame. `original` (already
maximally leaky per-frame) and the floor itself (nothing to amplify) show the
smallest Δ_redundancy, as expected if this is genuinely about pooling gains on
top of *existing* per-frame leakage.

### 5.2 Attacker-zoo Phase A, does synergy exist anywhere? `zoo_mamba_T16_s{1,2}.json`

Purpose: rule out two reasons synergy could be spuriously ~0 at T=3, (a) the
attacker's temporal receptive field being too small to see synergy even if
present, (b) the clip spanning too little real time for a trajectory to exist.
Tested with the *strongest* available attacker (`VideoMambaAttacker`, full-clip
reach, see §5.4) at the *longest* clip length (T=16) and two temporal spans.

| config | attacker reach | span | model | ordered AP | shuffled AP | **synergy** |
|---|---|---|---|---|---|---|
| T=16, stride=1 | full clip | 1.1s | oracle_FLOOR | 29.48 | 28.62 | +0.86 |
| T=16, stride=1 | full clip | 1.1s | original | 93.20 | 93.21 | **−0.01** |
| T=16, stride=2 | full clip | 2.1s | oracle_FLOOR | 29.72 | 30.15 | −0.43 |
| T=16, stride=2 | full clip | 2.1s | original | 93.43 | 92.97 | **+0.46** |

**Reading, decisive.** `original`'s synergy (−0.01, +0.46) sits *inside* the
floor's own noise band (+0.86, −0.43) at both spans. This was the maximally
favorable test, strongest attacker, longest window, most leakage-rich model
(undefended), and synergy is still indistinguishable from zero. Combined with
§5.1, **three independent configurations (different T, different attacker,
different span) all land at ~0 synergy.** Frame order does not appear to carry
recoverable trajectory information in this setting, at least at the horizons
tested (≤2.1s; longer spans untested, see §8.4).

**Consequence, thesis reframed.** "Temporal silhouette leakage" as originally
conceived (an attacker exploiting trajectory *order*) is not supported by the
evidence gathered so far. What *is* supported, and is a real, large,
non-obvious effect: **redundancy amplification**, a temporal attacker that
does nothing but pool per-frame estimates recovers dramatically more than a
per-frame attacker, and the amplification is largest precisely for the methods
that look most private under per-frame evaluation (ADVR, SalUn). This is
arguably a *more* useful and less obvious finding for the unlearning literature
than the original trajectory-synergy hypothesis would have been, because it
says something surprising about evaluation practice: **per-frame leakage audits
systematically understate true leakage for a class of methods, and the
understatement is worst for methods that are being sold as private.** See §8
for what this means for the method (ADVR-T/ADVR-TX need retargeting) and next
experiments.

### 5.3 Utility numbers (retained for reference, `benchmark.json`, pre-audit)

Not yet re-run under the corrected protocol's models; per-frame `leakage_metrics`
+ `relearn_audit` numbers from the original full pipeline run (baselines + ADVR +
ADVR-T), for context:

| model | person_IoU | retain_mIoU | blend_fail | silhouette_IoU | recover_pts |
|---|---|---|---|---|---|
| original | 81.2 | 33.5 | 96.5 | 33.2 | −26.6 |
| oracle | 0.0 | **30.8** | 52.6 | 16.8 | 9.5 |
| ADVR | 8.3 | 24.2 | 56.5 | 15.9 | 5.3 |
| ADVR-T | 1.4 | **22.5** | 54.3 | 14.6 | 7.7 |
| bl_DELETE_s1 | 0.0 | 17.1 | 73.0 | 16.8 | 13.1 |
| bl_FT_s1 | 0.0 | 17.4 | 77.5 | 18.7 | 12.8 |
| bl_NegGrad+_s1 | 0.0 | 4.9 | 92.3 | 16.1 | 0.0 |
| bl_NegGrad_s1 | 0.0 | 0.1 | 85.9 | 11.3 | 0.0 |
| bl_RL_s1 | 0.0 | 19.0 | 83.2 | 21.8 | 0.0 |
| bl_SCRUB_s1 | 0.0 | 0.1 | 85.9 | 11.3 | 0.0 |
| bl_SSD_s1 | 0.0 | 0.4 | 83.5 | 11.8 | 9.0 |
| bl_SalUn_s1 | 0.0 | 15.1 | 82.8 | 19.0 | 0.0 |

(`retain_mIoU` gap oracle-vs-ADVR-T: 30.8 vs 22.5, the utility confound of §4.4.)

### 5.4 Attacker architectures, reach comparison (`seg/attack.py`, measured)

Measured by perturbation: spike one frame's field, see which output frames
change.

| attacker | params | measured temporal reach | at T=21 |
|---|---|---|---|
| `TemporalSilhouetteNet` (conv3d, depth 3, kernel 3) | 0.44M | ±3 frames | **only 7/21 frames**, structurally blind to the rest |
| `ConvGRUAttacker` (bidirectional gated RNN) | 0.17M | ±10 (full clip) | full clip |
| `VideoMambaAttacker` (bidirectional selective SSM) | **0.08M** | ±10 (full clip) | full clip |

Mamba gets full-clip reach with **5× fewer parameters** than conv3d, so any gap
between conv3d and mamba results is attributable to receptive field, not
capacity. GPU cost measured: T=16, bs=2, mamba_ssm CUDA backend, **0.92GB
peak, 59ms/step**; cheap relative to the segmentation-model forward pass that
dominates wall-clock. `mamba_ssm==2.3.0`'s fused CUDA kernel confirmed working
(`selective_scan_cuda` imports, fwd+bwd runs on a toy `Mamba` layer); dependency-
free `S6Ref` fallback in `seg/mamba.py` validated for CPU/no-extension use.

`adv_temporal`'s `temporal_kind` argument was previously hardcoded to only
distinguish `"conv3d"` vs. everything-else-is-ConvGRU, passing `"mamba"` would
have silently trained against ConvGRU. Fixed to route through
`build_attacker(temporal_kind, ...)` in both `adv_temporal` and
`adv_temporal_x`.

### 5.5 Attacker zoo, Phase B, does ADVR-T suppress the real leak? (2026-07-27, complete)

Purpose: §5.1–5.2 established that the real leak is Δ_redundancy (pooling
gains), not synergy (order), and that ADVR (+15.9 to +19.2 across attackers)
amplifies dramatically more than the undefended model. Open question: does
ADVR-T's temporal-consistency term, designed and justified as an *order*
defense, do anything about Δ_redundancy at all, or does it leave the real leak
untouched while defending a mechanism that doesn't exist?

Ran conv3d / convgru / mamba × {oracle_FLOOR, original, ADVR, ADVR-T} at
T=16/stride=2 (2.1s span), `N_TRIALS=2`, `STEPS=500`, effective batch 4 clips
(`run_attacker_zoo.sh B`; `zoo_conv3d_T16_s2.json`, `zoo_convgru_T16_s2.json`,
`zoo_mamba_T16_s2.json`). ~13h wall-clock, zero errors.

**Δ_redundancy (= shuffled AP − T=3 frame AP) and synergy, all three attackers:**

| model | conv3d Δred / syn | convgru Δred / syn | mamba Δred / syn |
|---|---|---|---|
| oracle_FLOOR | −0.99 / +3.21 | +2.02 / −0.21 | +5.49 / −1.33 |
| original | +2.09 / +0.14 | +3.24 / +0.18 | +1.63 / −0.25 |
| ADVR | **+12.69** / +2.38 | **+19.21** / −1.43 | **+17.49** / +0.77 |
| ADVR-T | **+2.53** / +0.63 | **+5.43** / +0.60 | **+7.02** / −2.11 |

**Reading.**
1. **Synergy stays at noise level in all 12 cells**: no combination of model ×
   attacker shows synergy clearly separated from the floor's own synergy at
   that attacker. §5.2's conclusion (order carries no signal) survives the full
   zoo, including the strongest attacker.
2. **ADVR-T's Δ_redundancy is 2.4–5× smaller than ADVR's under every single
   attacker architecture** (ratio ADVR-T/ADVR: 0.20 conv3d, 0.28 convgru, 0.40
   mamba). This holds even against mamba, the attacker built specifically to
   remove the receptive-field excuse. ADVR-T is doing something real here, not
   an artifact of a weak attacker.
3. **The sharpest signal:** the floor's own Δ_redundancy *grows* with attacker
   strength (−0.99 → +2.02 → +5.49), even a model that never saw `person`
   leaks a little more to pooling as the attacker improves, which makes sense
   (richer context-prior features to average over). Under mamba, **ADVR-T's
   Δ_redundancy (+7.02) sits almost on top of the floor's own (+5.49)**, while
   ADVR's (+17.49) does not track the floor at all. ADVR-T's residual
   redundancy leak scales with the *irreducible* floor; ADVR's is a leak on top
   of the floor that stays large regardless of attacker.

**Interpretation, the mechanism, not just the number.** `_temporal_consistency`
(ADVR-T's `lam_tc` term) was originally motivated as an anti-*trajectory*
defense: cap the hole's frame-to-frame change so it can't look like an
independently-moving object. §5.2 showed that motivation was aimed at a
mechanism (order/synergy) that doesn't measurably exist here. But capping
frame-to-frame change magnitude has a second, unintended effect: it also caps
the magnitude of the *residual signal itself* available for an averaging
attacker to accumulate over T frames, which is exactly what Δ_redundancy
measures. **ADVR-T works, but for a different reason than it was designed
for**: variance-capping against pooling, not exchangeability against ordering.
This is a cleaner and more defensible story for the method section than the
original one, and it means `ADVR-TX`'s `OrderDiscriminator` (built to attack
order specifically) is very likely solving a problem that isn't there, see the
updated queue below.

### 5.6 `lam_ord` ablation (2026-07-29, complete)

Purpose: item 2a. Trained `ADVR-TX0` (`lam_ord=0`, local-ring `L_tc` only) and
`ADVR-TX` (`lam_ord=1.0`, full method) with otherwise identical config
(matching ADVR-T's exactly: epochs=4, `lam_tc=0.5`, `temporal_kind=conv3d`,
bs=2, clip_len=3, confirmed from ADVR-T's saved checkpoint config). Fixed a
real bug first: `test_candidate.py`'s `is_temporal` check only recognized
`"ADVR-T"`, so `"ADVR-TX"` fell into the per-frame branch and would have fed
single-frame batches into a method expecting clips, fixed to
`args.name in ("ADVR-T", "ADVR-TX")`, plus added the missing `--lam_ord` CLI
flag and `"mamba"` to `--temporal_kind`'s choices. Verified with a live 45s
smoke run before committing the full ~28h job.

Training times: `ADVR-TX0` 852.4min, `ADVR-TX` 860.6min, both close to
ADVR-T's 822.2min; `adv_temporal_x` always trains `OrderDiscriminator`
regardless of `lam_ord` (only the generator's use of it is gated), so the
~4–5% overhead is present in *both* arms, not just `lam_ord=1.0`.

Eval: mamba, T=16, stride=2 (identical protocol to §5.5), `N_TRIALS=2`,
re-scoring `oracle_FLOOR`/`original`/`ADVR`/`ADVR-T` fresh in the same run for
direct comparability (`zoo_mamba_T16_s2_ablation.json`).

| model | ordered AP | shuffled AP | synergy |
|---|---|---|---|
| oracle_FLOOR | 28.04 ± 1.13 | 29.68 | −1.64 |
| original | 93.30 | 93.17 | +0.13 |
| ADVR | 48.46 | 45.00 | +3.46 |
| ADVR-T | 20.65 | 23.40 | −2.75 |
| ADVR-TX0 (`lam_ord=0`) | 24.81 ± 0.36 | 24.58 | +0.23 |
| ADVR-TX (`lam_ord=1.0`) | 22.70 ± 2.00 | 22.92 | −0.22 |

**Reading, item 2a is inconclusive, not negative.** `lam_ord=0→1.0` moves
ordered AP from 24.81 to 22.70 (−2.11), but ADVR-TX's own trial std at
N_TRIALS=2 is **2.00**, the observed effect is within one std of noise. This
does not demonstrate `lam_ord` is inert; it demonstrates the current experiment
(2 trials) cannot distinguish "inert" from "small positive effect." Synergy
stays near zero for both arms (+0.23, −0.22) either way, consistent with
§5.2/§5.5, no order signal exists for `OrderDiscriminator` to remove, which is
itself evidence for cutting it (paying training complexity for an
undetectable effect), just not yet a clean ablation result.

**Unplanned finding, flagged rather than resolved:** both ADVR-TX variants
leak *more* than plain ADVR-T under this protocol (24.81, 22.70 vs. ADVR-T's
20.65), i.e. the local-ring `L_tc` (designed as a strict improvement over
ADVR-T's global-mean reference, §3.3) underperforms the global-mean version
here. Two explanations not yet distinguished: (a) a genuine regression, a
per-hole local reference may be diluted when the hole's own neighborhood
happens to be high-motion, where a global reference would have capped it
harder; (b) ordinary run-to-run variance, each of ADVR-T/ADVR-TX0/ADVR-TX is
a single training run with no shared seed control. **Also note:** ADVR-T
(20.65) and ADVR-TX (22.70) both sit *below* the oracle floor (28.04), per the
still-open utility confound (§4.4), this is far more likely to be retain_mIoU
damage (less information of any kind to extract) than genuinely-better-than-
optimal privacy. Neither `ADVR-TX0` nor `ADVR-TX` has had `run_benchmark.py`
run against it yet, that reading is the fastest next step (§8, revised below)
before spending more GPU time on trial-count variance for either the
`lam_ord` question or the local-ring-vs-global-mean question.

---

## 6. Compute log

### 6.1 Reference + baseline training (full VSPW, real GPU time)
- 8 baselines (FT, NegGrad, NegGrad+, RL, SalUn, SSD, SCRUB, DELETE): SCRUB
  longest at 390.0min (~6.5h), DELETE 256.8min; others faster. All completed and
  checkpointed.
- ADVR (per-frame, 4 epochs, 138,076 frame-units): **1052.8min (~17.5h)**, far
  longer than any baseline, consistent with training both the segmentation
  model and an adversarial discriminator each step.
- ADVR-T (temporal, 4 epochs, clip_len=3, 45,976 clip-units): **822.2min
  (~13.7h)** on retry (first attempt OOM'd, see §6.2).

### 6.2 Known failure modes (for anyone re-running this pipeline)
- **bs=8 OOM** on the 16GB card for per-frame training at 512 res /
  DeepLabV3-R50, use bs=6 (project1's proven setting) +
  `PYTORCH_ALLOC_CONF=expandable_segments:True`.
- **ADVR-T's first attempt OOM'd immediately**: default `--bs 6` for clips
  means 6 clips × 3 frames = 18 frames/batch through *two* models (segmentation
  + temporal discriminator), 3× the per-batch load of per-frame ADVR. Fixed:
  `--bs 2` for clip-based training (2×3=6 frames/batch, matching ADVR's
  per-batch frame count).
- **`nohup ... &` + `disown` does not survive session/tool teardown** on this
  machine; `setsid` does, but **does not survive a full OS reboot** (one
  occurred mid-pipeline, 2026-07-23; 6/8 baselines had already checkpointed and
  survived, only the in-progress SCRUB run was lost and retrained from scratch
 , the skip-if-exists resumability in `run_baselines.py`/`run_all.sh` worked
  correctly).
- **`run_all.sh` lacked `set -o pipefail`**: `$PY script.py | tee -a log`
  reports `tee`'s exit code to `set -e`, not the script's, so a crashed Python
  stage (ADVR-T's OOM) let the pipeline silently continue into the next stage.
  Fixed.
- **Person-frame filtering was rescanned from scratch (~8min, single-threaded)
  by every pipeline stage** independently, now cached
  (`_filter_person_cached`, mirrors `VSPWClip`'s clip-index cache).

---

## 7. Script inventory, current vs. superseded

| script | status | purpose |
|---|---|---|
| `scripts/train_seg.py` | current | trains `original`/`oracle` reference models |
| `scripts/run_baseline.sh`, `run_baselines.py` | current | 8 published baselines |
| `scripts/test_candidate.py` | current | trains ADVR / ADVR-T / ADVR-TX |
| `scripts/run_benchmark.py` | current | per-frame utility + leakage table |
| `scripts/eval_ckpt.py` | current | ad hoc single-checkpoint eval |
| `scripts/run_all.sh` | current | full pipeline orchestrator (pipefail-fixed) |
| `scripts/run_temporal_attack.py` | **SUPERSEDED, do not use for new results** | old T-sweep protocol, has all 3 defects in §4.1–4.3 |
| `scripts/sweep_clip_len.sh` | **SUPERSEDED, results invalid** | drove the confounded T-sweep |
| `scripts/run_temporal_protocol.py` | **current, canonical** | corrected protocol: anchored clips, matched attacker budget, threshold-free scoring, ordered/shuffled synergy arms |
| `scripts/sweep_protocol.sh` | current | drives `run_temporal_protocol.py` across T (conv3d only, see caveat below) |
| `scripts/run_attacker_zoo.sh` | current | attacker-architecture × temporal-span sweep, Phase A (done, §5.2) / Phase B (done, §5.5) |
| `scripts/run_advrtx_ablation.sh` | current | trains ADVR-TX0/ADVR-TX (`lam_ord` ablation), evals both under mamba/T=16/s2 (done, §5.6) |

**Caveat on `sweep_protocol.sh`:** its T=2/5/8/16 arms (beyond the completed
T=3) use the conv3d attacker, which per §5.4 cannot see more than ±3 frames,
those arms were stopped before running (2026-07-27) once §5.2's Mamba result
made clear that conv3d-based long-T arms cannot support any claim about
long-range structure. Only `protocol_T3.json` from this sweep is valid and
used above; do not resume the conv3d T=2/5/8/16 arms without switching them to
`TEMPORAL_KIND=mamba` first.

**Result files, validity:**
- Valid / current: `protocol_T3.json`, `zoo_mamba_T16_s1.json`, `zoo_mamba_T16_s2.json`,
  `zoo_conv3d_T16_s2.json`, `zoo_convgru_T16_s2.json`, `zoo_mamba_T16_s2_ablation.json`.
- Superseded, kept for audit trail only, **do not cite**: `temporal_attack.json`,
  `temporal_attack_T{2,5,8,16}.json` (old protocol, all of §4.1–4.3 apply).
- Independent of the temporal-protocol bugs, still valid: `benchmark.json`
  (per-frame utility/leakage; subject to the utility confound in §4.4).

---

## 8. Bug audit (2026-07-30), two code-level bugs found and fixed

A dedicated pass over every formula and data-pipeline step (not just protocol
design, actual code correctness), prompted directly by a request to hunt for
lingering logical/mathematical errors. Found two, both real, both now fixed;
scope of impact assessed for each rather than assumed.

### 8.1 Biased std estimator in `run_temporal_protocol.py`'s `agg()`

`agg()` computed `np.std(vals)`, which defaults to `ddof=0` (population std).
These values are a **sample** of `N_TRIALS` independent training/eval runs
drawn from a population of unknown variance -- the correct estimator is the
unbiased sample std, `ddof=1`. At `N_TRIALS=2` (used throughout every sweep to
date), `ddof=0` understates the true sample std by a factor of
**sqrt(2) ~= 1.414x** (verified numerically: two arbitrary values gave std 1.055
at ddof=0 vs 1.492 at ddof=1, ratio exactly sqrt(2)). This directly affects:

- Every `_std` figure reported in `protocol_T3.json`, all `zoo_*.json` files,
  and `zoo_mamba_T16_s2_ablation.json` -- all ~41% too small.
- The `"leaks iff AP > floor_mean + 2*floor_std"` threshold in
  `run_temporal_protocol.py`'s summary block -- every LEAK/safe verdict
  computed so far used a threshold that was too tight (too easy to call LEAK).
- The §5.6 ablation conclusion specifically: ADVR-TX's reported std (2.00) was
  under the biased estimator; the correct value is ~2.83, which makes the
  "inconclusive, within one std of noise" reading in §5.6 even more clearly
  correct (the gap the ablation was testing is smaller relative to noise than
  originally stated, not larger) -- **the qualitative conclusion of §5.6 does
  not change, but the quantitative noise band understated in that section
  should be read as ~41% wider than written.**

**Fixed:** `agg()` now uses `ddof=1` when `N_TRIALS>1` (falls back to `ddof=0`
only for a single sample, where `ddof=1` is undefined). Not yet re-run against
the existing JSONs -- the effect sizes reported so far (Phase B's Δ_redundancy
gaps in particular, e.g. ADVR's 12.69-19.21 vs. ADVR-T's 2.53-7.02) are large
enough relative to a 41% wider noise band that the qualitative story in §5.5
is not expected to change, but this has not been explicitly re-verified.

### 8.2 Crop-before-merge label corruption in `VSPWUnlearn` (data pipeline)

**The more serious of the two.** `VSPWUnlearn.__getitem__` built the "orig"
label via `VSPWSeg.__getitem__` (which applies VSPWSeg's random crop/zoom
augmentation internally), and only THEN ran `merge_person_label` on the
*already-cropped* label to produce "merged" -- the dominant-neighbour
relabeling target that unlearning methods train the forget region toward.
`VSPWClip` (used by ADVR-T/ADVR-TX) never had this bug: it already computed the
merge on the full-resolution label before applying a shared crop geometry to
image/orig/merged.

**Why the ordering matters, demonstrated, not just argued:** `merge_person_label`
does a nearest-non-person-non-ignore-pixel fill via distance transform. If the
region it's given contains NO valid non-person pixel, it has nothing to grab
and **silently returns the label unchanged**. A crop taken after a random
zoom-in (this augmentation samples scale in [0.5, 2.0]x before cropping to the
target size) can plausibly isolate an all-person region. Reproduced directly:
a synthetic all-person 64x64 label fed through `merge_person_label` returns
itself unchanged -- i.e. the "forget" target for that sample still says PERSON,
so the unlearning loss for that batch element trains the model *toward*
keeping the class on exactly the crop where forgetting matters most.

**Measured on real data, not just the synthetic worst case** (400-crop and
300-crop samples from the real `train` split, `seg/data.py`, size=512):
- The literal *fully-degenerate* no-op (merged == orig, zero fill possible)
  did not occur in 400 sampled real training crops -- VSPW's person instances
  evidently rarely fill an *entire* 512x512 crop even under 2x zoom, so the
  worst-case failure mode, while real and reproducible, is rare on this
  dataset specifically.
- The subtler, more common failure **does occur**: comparing the buggy
  (crop-then-merge) vs. correct (merge-then-crop) label using the *same*
  sampled crop geometry, **on average 5.5% of person-region pixels resolve to
  a different (wrong) fill class**, with 16.7% of crops showing >10%
  disagreement and a worst case of 29.5%. This is real, non-trivial label
  noise in the training target for every affected method, not a purely
  theoretical risk.

**Blast radius -- narrower than the failure mode suggested, traced exactly
rather than assumed.** Grepped every consumer of `VSPWUnlearn`'s `merged`
field:
- All 8 published baselines, as actually invoked by `scripts/run_baselines.py`
  (`seg/baselines/*.py`), **discard `merged` entirely** (`_merged` in every
  loader unpacking) -- FT trains on retain-only CE against `orig`, NegGrad/
  NegGrad+/RL/SalUn/SSD/SCRUB/DELETE all work from `orig` + forget/retain masks,
  never touching the merged label. **None of the 8 baseline checkpoints are
  affected.**
- `seg/methods.py::ft` does consume `merged` (line ~53), but `seg/methods.py`'s
  training-loop functions (`ft`, `ga`, etc.) are never imported by any script
  that trains anything -- only three of its *utility* functions
  (`_clone`/`_masked_ce`/`_masks`) are imported, by `seg/candidates.py`. This
  code path is dead for every result produced so far.
- `seg/candidates.py::adv_replay` (**plain ADVR**) is the ONLY trained method
  that consumes `merged` from `VSPWUnlearn` with `crop=True`
  (`_masked_ce(z, merged, forget)`). ADVR-T/ADVR-TX are unaffected: their main
  loader is `VSPWClip` (already correct order) and their retain-replay loader
  is `VSPWSeg(target="orig")` (no merge involved at all).

**Net scope: exactly one already-trained checkpoint is affected --
`checkpoints/unlearned_ADVR.pth`** (the ~17.5h / 4-epoch / 138k-frame-unit
run), which is the "ADVR" reference point in every comparison in §5.1, §5.2,
§5.5, and §5.6. Every ADVR-T, ADVR-TX0, ADVR-TX, and all 8 baseline numbers in
this document stand as computed with respect to this bug.

**Fixed:** `VSPWSeg` refactored to expose reusable `_sample_geom()`/`_apply()`
(mirrors `VSPWClip`'s existing pattern; `VSPWSeg.__getitem__`'s own behavior is
unchanged, verified). `VSPWUnlearn.__getitem__` rewritten to load the raw
(full-resolution) label via `self.base._load(idx)`, run `merge_person_label`
on it BEFORE any cropping, then apply one sampled geometry to img/orig/merged
together -- exactly `VSPWClip`'s pattern. Verified: 0/60 real val-split samples
at `crop=True` now leave any PERSON pixel in the merged label (previously
demonstrated possible); `VSPWSeg`'s own `target="merged"` behavior (which was
already correct) is provably unchanged by the refactor.

**Not yet done, flagged for a decision rather than auto-launched given the
compute cost:** retraining `unlearned_ADVR.pth` under the fix (~17.5h based on
the original run) would be needed for a fully clean ADVR number. Given the
measured average disagreement (5.5%) is real but moderate, and ADVR's loss has
two other terms (retain-KL anchor, adversarial-fool) providing signal
alongside the corrupted blend term, the qualitative Δ_redundancy story (ADVR
amplifying far more than ADVR-T, tracked across three attacker architectures
in §5.5) is not expected to reverse -- but the exact magnitude of ADVR's
numbers throughout this document should be treated as measured on a
mildly-noisy-target checkpoint until it is retrained and re-benchmarked.

## 9. Mechanistic diagnosis of the ADVR-TX regression (2026-07-30)

§5.6's ablation left an unexplained anomaly: ADVR-TX0/ADVR-TX were worse than
plain ADVR-T on **both** utility (retain_mIoU 20.9/21.5 vs. ADVR-T's 22.5,
measured via `scripts/eval_ckpt.py` on the two new checkpoints -- the
"cheap 2c" utility check) **and** privacy (mamba ordered AP 24.81/22.70 vs.
ADVR-T's 20.65). A simultaneous regression on two metrics that usually trade
off against each other is the signature of a broken training incentive, not
coincidence -- traced with autograd rather than guessed at.

**Isolating the culprit.** `ADVR-TX0` (`lam_ord=0`) has no order-adversary
pressure on the generator at all (the `if lam_ord>0` block never executes), so
its generator loss is *identical in composition* to ADVR-T's, with exactly one
substitution: local-ring `L_tc` in place of global-mean `L_tc`. Since
`ADVR-TX0` alone already reproduces the double regression, the local-ring
reformulation is the cause, not the order adversary or the extra discriminator.

**Root cause 1 -- reward hacking via a differentiable reference.** v1's local
reference (`ref`) was computed from the model's own live field with no
stop-gradient, so `relu(dO_hole - ref)` had two ways to shrink: suppress the
hole's motion (intended), or inflate `ref` by increasing motion in the ring of
pixels *around* the hole (a shortcut). Verified directly by backprop on a
synthetic clip: v1 puts nonzero gradient on ring pixels outside the hole
(0.16 relative units) that v2 (after fixing) reduces to exactly 0.0. Quantified
why the global-mean version doesn't suffer from the same defect in practice
despite having the same differentiable-reference structure: averaged over
~4000+ retain pixels, its equivalent exploit gradient is **~1.9 million times
smaller per pixel** than the hole's own gradient (measured) -- never worth
taking. Concentrated over a ~9x9 ring (~50-70 pixels), it was cheap, and the
optimizer took it: instead of truly suppressing the hole (which would fix
privacy) or leaving nearby retain predictions alone (which would preserve
utility), the model nudged the boundary region -- corrupting retain pixels
near the person (worse utility) while leaving the hole's real signal untouched
(worse privacy). This explains the double regression precisely, not just one
side of it.

**Root cause 2 -- excess permissiveness near dynamic backgrounds.** Independent
of the gradient-hacking issue, a purely local reference can also just be
*correct* to compute yet still a bad idea: under camera motion or any busy
local context, the ring's own average change is large, handing the hole a
generous "free" motion budget exactly where suppression matters most.

**Fix (`seg/candidates._temporal_consistency_local`, v2), both parts
independently verified before being wired into the method:**
1. `.detach()` the field used to compute the reference (`dO_det`) -- `ref`
   becomes a frozen target the hole must match, not a lever the optimizer can
   pull. Verified: ring gradient now exactly 0.0 on the same synthetic case
   that showed 0.16 before, hole gradient unchanged and still positive.
2. `ref = min(local_ref, global_ref)` -- the hole is never handed more room
   than the stable global budget would allow, and only gets a *stricter*
   (smaller) one when its local context happens to be unusually static, which
   was the actually-useful half of the original idea. Verified: quiet-hole
   probe still returns ~0, moving-hole probe still returns a clearly larger
   value (0.0 vs. 0.0143 in the same synthetic setup used to validate v1).

Full `adv_temporal_x` training step re-validated end-to-end with v2 wired in:
all params update, stays finite, T=1 degenerate clip doesn't crash. **Not yet
trained** -- `ADVR-TX0`/`ADVR-TX` were both trained against v1 and their
checkpoints/numbers in §5.6 reflect the bug; a clean read of whether the
local-ring idea has any merit requires retraining against v2, which is the
natural next experiment (see queue below).

## 9b. ADVR-P: attacking the mechanism directly (2026-07-30/31, trained + evaluated)

Built two new pieces, both designed specifically to attack Δ_redundancy (the
confirmed real leak) rather than order (confirmed not a leak) or a magnitude
proxy, and specifically designed to avoid the reward-hacking failure mode found
in the local-ring `L_tc`:

- **`PoolDiscriminator`** (`seg/attack.py`) -- sees ONLY `O.mean(dim=1)`,
  collapsed before any learned parameter touches it. Verified exactly
  permutation-invariant (4e-6 float-noise max diff on a real shuffle test),
  correct output shape, gradient distributes uniformly across all T input
  frames (matches the exact math of `mean`'s backward, not approximate).
- **Two-sided `L_tc`** (`_temporal_consistency_local(..., two_sided=True)`) --
  replaces `relu(dO-ref)` with `(dO-ref)^2`, using the already-fixed
  detached/min(local,global) reference. Verified the specific defect it closes:
  a frozen (temporally-invariant) hole scores exactly `0.0` under the one-sided
  version (mathematically blind to it) but a clearly positive value under
  two-sided. Reconfirmed the ring-gradient-exploit fix still holds (0.0) under
  the new formula.
- **`ADVR-P`** (`seg/candidates.adv_pool`) -- combines both, keeps `DT` (the
  usual reconstruction discriminator) rather than replacing it, drops
  `OrderDiscriminator` entirely. Validated end-to-end on 5 scenarios (normal,
  no-replay, `lam_tc=0`, degenerate `T=1`, degenerate no-person-in-clip) before
  training; also confirmed live on real GPU/data for 50s with no errors.

**Trained** (`scripts/run_advrp.sh`, ADVR-T's exact recipe -- epochs=4,
clip_len=3, bs=2, lam_r=1.0, lam_adv=1.0, lam_tc=0.5 -- plus `lam_pool=1.0`):
853.5min (vs. ADVR-T's 822.2min, ~3.8% overhead from the extra discriminator,
in line with ADVR-TX's ~3.7% overhead from its extra adversary).

**Evaluated** under the established mamba/T=16/stride=2 protocol
(`results/seg/zoo_mamba_T16_s2_advrp.json`, re-scoring the full reference set
fresh in the same run for direct comparability) plus a utility check
(`scripts/eval_ckpt.py`):

| model | retain_mIoU | ordered AP (mamba) |
|---|---|---|
| oracle | 30.8 | 30.82 ± 0.14 |
| ADVR | 24.2 | 49.15 ± 2.25 |
| ADVR-T | 22.5 | 21.28 ± 2.34 |
| ADVR-TX0 | 20.9 | 24.81 ± 0.36 |
| ADVR-TX | 21.5 | 22.70 ± 2.00 |
| **ADVR-P** | **21.2** | **18.79 ± 1.23** |

**Reading -- a real tradeoff, not a clean win, and not a regression either.**
Unlike `ADVR-TX` (worse than ADVR-T on BOTH axes simultaneously -- a dominated
point, caused by the reward-hacking bug in §9), `ADVR-P` moves ALONG the
frontier: privacy improves by ~2.5 AP points (18.79 vs. 21.28) at a utility
cost of ~1.3 mIoU points (21.2 vs. 22.5). This is exactly the qualitative
direction the mechanistic design predicted (attack Δ_redundancy directly
instead of through a magnitude proxy), and `ADVR-P`'s own trial-to-trial std
(1.23) is tighter than `ADVR-T`'s (2.34) at the same `N_TRIALS=2` -- a mildly
encouraging secondary signal. **Not yet a confirmed result**: both the privacy
gap (2.5 points) and the utility gap (1.3 points) are small relative to what a
single training seed and `N_TRIALS=2` can resolve with confidence. Needs
(a) more trials on the privacy side and (b) a second training seed before the
magnitude of this tradeoff -- or whether it is a tradeoff at all rather than
noise -- can be trusted for a table meant for the paper.

## 10. Next experiments (live queue, update as completed)

0. **[NEW, top priority] Confirm or reject the ADVR-P tradeoff.** §9b found
   ADVR-P moves along the privacy-utility frontier relative to ADVR-T
   (+2.5 privacy points, -1.3 utility points), but both gaps are small relative
   to single-seed/`N_TRIALS=2` noise. Two things needed before trusting this:
   (a) raise `N_TRIALS` to 5-10 for ADVR-T and ADVR-P specifically under
   mamba/T=16/stride=2 (cheap -- no retraining, just more attacker trials
   against the existing checkpoints); (b) train a second seed of ADVR-P (and
   ideally ADVR-T) to check the utility/privacy numbers aren't an artifact of
   one particular training run. If the gap survives both, ADVR-P is a genuine,
   reportable tradeoff point; if it doesn't, the honest conclusion is that
   ADVR-T and ADVR-P are statistically indistinguishable and the added
   complexity (PoolDiscriminator, two-sided L_tc) isn't earning its keep.
1. ~~**Phase B of the attacker zoo**~~, **DONE 2026-07-27, see §5.5.** Result:
   ADVR-T's Δ_redundancy is 2.4–5× smaller than ADVR's under every attacker
   (conv3d/convgru/mamba), and under mamba specifically it tracks the
   irreducible floor's own Δ_redundancy almost exactly (+7.02 vs +5.49) while
   ADVR's does not (+17.49, flat regardless of attacker). ADVR-T genuinely
   suppresses the real leak (redundancy), via a mechanism (variance-capping)
   different from the one it was designed around (order-scrambling).
2. ~~**`lam_ord` ablation (2a)**~~, **DONE 2026-07-29, see §5.6. Inconclusive,
   not negative.** `lam_ord=0→1.0` moved ordered AP 24.81→22.70 (Δ=−2.11),
   within one std of ADVR-TX's own trial noise at N=2 (std 2.00). Synergy near
   zero either way, consistent with order not being the leak, evidence for
   deprioritizing `OrderDiscriminator`, not yet a clean statistical result.
   **Unplanned finding to chase first (2c below) before re-running this at
   higher N:** both ADVR-TX variants leaked *more* than plain ADVR-T in this
   run (24.81, 22.70 vs. 20.65), the local-ring `L_tc` may be a regression
   vs. ADVR-T's global-mean version, or this is just single-run seed noise.
2c. ~~**Utility check on ADVR-TX0/ADVR-TX**~~, **DONE 2026-07-30**
   (`scripts/eval_ckpt.py`, no retraining needed). Result: both are WORSE than
   ADVR-T on retain_mIoU (20.9, 21.5 vs. 22.5), not just leakier, a
   simultaneous regression on both axes, which is not what "just more damage"
   alone would predict (more damage should trade utility for privacy, not lose
   on both). This pointed at a broken training incentive rather than a fair
   privacy-utility tradeoff, and led directly to §9's diagnosis.
2d. ~~**Mechanistic diagnosis + fix (was 2b's question, answered by tracing
   gradients instead of designing a new term)**~~, **DONE 2026-07-30, see §9.**
   Found via autograd, not guessed: v1's local-ring `L_tc` let the model reduce
   the loss by inflating motion in the ring AROUND the hole (reward hacking via
   an undetached reference) rather than truly suppressing the hole's own
   motion, corrupting nearby retain predictions (worse utility) while leaving
   the hole's real signal untouched (worse privacy), simultaneously. Fixed
   (`.detach()` the reference; `ref = min(local, global)`), verified
   independently (ring gradient 0.0, hole gradient unchanged, quiet/moving-hole
   discrimination preserved) and end-to-end (`adv_temporal_x` still trains,
   stays finite, T=1 doesn't crash). **2b (an anti-correlation-based
   alternative objective) is deprioritized** below the more urgent question:
   does the FIXED local-ring idea (v2) actually beat plain global-mean L_tc, or
   was the entire "spatially-aware reference" direction not worth pursuing even
   without the bug? Needs a retrain to answer, see item 2e.
2e. **[NEW, top priority] Retrain ADVR-TX under v2 and re-run the mamba
   ablation.** `ADVR-TX0`/`ADVR-TX`'s existing checkpoints were trained against
   the buggy v1 local-ring `L_tc`, their numbers throughout §5.6/§9 do not
   reflect the fix. Retrain both (same command as `run_advrtx_ablation.sh`,
   ~13-14h each based on the original run) and re-eval under mamba/T=16/
   stride=2 (matching §5.5/§5.6's protocol) to see whether v2 closes the gap to
   ADVR-T on utility, and whether it now BEATS ADVR-T on Δ_redundancy (the
   original motivation for the local-ring idea, a hole in a static region
   should get a stricter budget than the global average gives). If v2 still
   doesn't beat plain ADVR-T, the honest conclusion is that ADVR-T's simple
   global-mean formulation is the right level of complexity for this method and
   the local-ring direction (and 2b's alternative) should be dropped, not
   iterated on further.
3. **Matched-utility comparison** (§4.4, still open, 2c above is the narrow,
   fast version of this for just the two new checkpoints; this item is the
   full sweep). Sweep `lam_r` / replay weight per method so leakage is compared
   at fixed `retain_mIoU`, or report a full privacy-utility Pareto frontier.
   Without this, part of "ADVR-T leaks less" is still confounded with "ADVR-T
   is more damaged" (retain_mIoU 22.5 vs. oracle's 30.8), §5.5's finding
   survives this concern somewhat (it compares Δ_redundancy, a *relative*
   quantity, and ADVR-T's Δ tracking the floor is a shape argument more than a
   magnitude one) but the confound should still be closed properly before
   publication.
4. **N-trial variance on the headline redundancy numbers.** Still `N_TRIALS=2`
   throughout §5.5–5.6. Given this is now the paper's central quantitative
   result, raise to 5–10 (project1's convention), prioritize the mamba/T=16/
   stride=2 cells for ADVR and ADVR-T specifically (the headline comparison),
   then ADVR-TX0/ADVR-TX once 2c has ruled out a pure utility-damage
   explanation for their numbers.
5. **Generalize beyond `person`.** Test whether the redundancy-amplification
   pattern (large for per-frame-private methods, small for undefended/floor,
   suppressed by ADVR-T) holds for a second forget class with different motion
   statistics (e.g. `car`), to rule out a person-specific artifact.
6. **Longer-horizon synergy check (stretch).** §5.2/§5.5 tested spans up to
   2.1s; still open whether synergy appears at longer real-time spans (5s+,
   `frame_stride` 4–8 with a correspondingly raised `min_video_len`, accepting
   the resulting drop in eligible videos, would need to re-verify the
   anchor-identity invariant at that higher stride). Lower priority than 2c–4
   now that redundancy (not synergy) is the established headline mechanism.
