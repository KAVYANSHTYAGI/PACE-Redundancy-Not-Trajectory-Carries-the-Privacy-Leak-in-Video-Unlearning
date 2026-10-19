# Positioning: Temporal Silhouette Leakage (video extension of project1)

Started 2026-07-20. Direct extension of `machine_unlearning/project1`'s spatial
silhouette leakage thesis into video semantic segmentation. See
[[pe-isp-research-direction]] and [[mu-project1-direction]] for the parent line.

## 0. The design fork (resolved)

Video "unlearning" splits into two structurally different problems:

- **(A) Class-level**, video semantic/panoptic segmentation, where the forget
  target is a fixed semantic category (e.g. "person") present across all clips.
  This is a mechanical extension of ADVR: same oracle-floor logic, same
  discriminator idea, just add a temporal receptive field.
- **(B) Instance-level**, memory-bank models (SAM2, XMem, Track-Anything), where
  objects are arbitrary, per-video, user-prompted entities with no fixed class.
  "Unlearn person" doesn't parse there, the right formalization is closer to
  "scrub this tracked instance's memory embedding so a future clip with the same
  person is no longer followed." Needs its own problem definition from scratch,
  and the reference models (SAM2 etc.) are far larger than anything trainable end
  -to-end on a single 16GB/32GB card.

**Decision: (A) is the primary thesis.** It reuses ~80% of project1's codebase and
conceptual apparatus, is tractable on the current compute (RTX 5080 local, 5090
over SSH, same envelope that trained DeepLabV3-R50 on ADE20K), and is a real,
submittable paper on its own. (B) is noted as explicit future work / a short
"broader implications" pilot, not the main experiment, it is the more novel but
much more expensive direction, and shouldn't block a submittable result.

## 1. Landscape (verified 2026-07-20, not just recalled)

Fact-checked via live search before committing to the novelty claim, do not trust
the claim below without the citations, they were the point of checking:

- **"Temporal Unlearnable Examples"** (arXiv 2507.07483, ICCV 2025, Wu et al.) is
  real and does exactly what it says: generates imperceptible perturbations so
  visual object trackers trained on the perturbed video *fail to learn* it. This
  is **data poisoning to prevent training**, not post-hoc removal from an already
  -trained model. Different problem; cite it as the closest privacy-adjacent prior
  art, not as prior unlearning work.
- **"Hierarchical Visual Prompt Learning for Continual Video Instance
  Segmentation"** (ICCV 2025) uses "forgetting" in the **catastrophic-forgetting**
  sense (losing old classes while learning new ones), the opposite motivation
  from privacy-driven deliberate unlearning. **Trap for related work**: a reviewer
  skimming citations could conflate the two. The paper needs an explicit paragraph
  distinguishing "unlearning" (deliberate, privacy-motivated, forget signal is
  known and given) from "catastrophic forgetting" (accidental, capacity-driven, no
  forget signal), this distinction is not obvious to a reader coming from the
  continual-learning side.
- **CVPR 2026 "Machine Unlearning for Vision" (MUV) workshop** call-for-papers
  explicitly lists video *classification* and video *generation* under its scope,
  but **not** video segmentation/temporal unlearning, confirms the niche is
  uncovered even in the one venue built for this exact topic. Its March 2026
  deadline already passed for this cycle; track the next iteration as a venue
  target alongside AAAI/CVPR/ICCV main tracks.
- No paper found doing post-hoc class or instance removal from a trained video
  segmentation/tracking model for privacy, under any of: "video unlearning",
  "forget class video segmentation", "SAM2 memory privacy leakage", "temporal
  silhouette leakage". The gap the user's brainstorm identified is real as of this
  search, reconfirm this before final submission in case something lands between
  now and then (the area is moving fast: 3 of the 4 cited "adjacent" ICCV papers
  are from ICCV 2025, i.e. the last few months).

## 2. The sharper hypothesis (temporal analog of project1 HYP)

Project1's HYP: forgetting in dense prediction = driving $I(M;O)\to 0$ (spatial
mutual information between the forgotten mask and the model's output field),
not merely re-labeling.

**Temporal HYP**: a video segmentation model that correctly suppresses a person
label in every frame can still encode the person's **trajectory**, its
identity, motion, and occlusion pattern, in the *sequence* of output fields, even
when no single frame leaks. Formally, let $O_t = O_\theta(x_t)$ be the per-frame
output field and $M_t$ the true forget-mask at frame $t$. Project1 shows
$I(M_t; O_t) \to 0$ is achievable per-frame (ADVR). The open question is whether
$I(M_{1:T};\,O_{1:T}) \to 0$ **jointly**, i.e. whether a temporal attacker with a
window over $O_{1:T}$ can recover the trajectory even when every marginal
per-frame attacker fails. This is the thing no per-frame image method (including
ADVR applied frame-by-frame) is designed to defend against, and it is the paper's
actual contribution target: **temporal silhouette leakage is a strictly harder
and distinct failure mode from spatial silhouette leakage, not just "the same
thing averaged over frames."**

## 3. Concrete, compute-feasible experimental plan

Mirrors project1's structure exactly so the harness/conventions transfer:

- **Dataset**: VSPW (CVPR 2021), 3,536 videos, 251,633 frames, 124 classes,
  43 GB, ungated Google Drive download. Direct video analog of ADE20K (same
  taxonomy lineage); "person" is one of the 124 classes. **Not yet on disk,
  first concrete task is downloading it.**
- **Primary model**: DeepLabV3-R50 (reuse project1's `seg/model.py`) + a
  lightweight temporal propagation module (small ConvGRU or optical-flow-warped
  feature fusion over a 3-5 frame window) trained on VSPW. Deliberately *not*
  SAM2/XMem/Mamba-based (TV3S) as the primary model, those are architecturally
  heavier and the point of the paper is the leakage phenomenon, not
  reproducing a foundation model. A **backbone-generality check** with a heavier
  temporal architecture (e.g. TV3S-style state-space sharing) is the natural
  "SegFormer-B2" analog from project1, a secondary experiment, not the headline.
- **Forget task**: unlearn "person" across a clip. Oracle = retrain from scratch
  with person pixels dominant-neighbour-relabelled in *every frame independently*
  (direct per-frame extension of project1's eq. 1, no new relabelling logic
  needed, since VSPW gives dense per-frame masks, not per-video object prompts).
- **Attacker (the new part)**: a **temporal discriminator** with a window over
  $k$ consecutive output fields (e.g. 3D-conv or ConvGRU over stacked $O_{t-1:t+1}$),
  trained to reconstruct the person mask *trajectory*, not a single frame. This is
  the direct temporal generalisation of project1's `SilhouetteNet`/U-Net/ViT zoo,
  same zoo-of-architectures rebuttal-proofing strategy should be built in from the
  start this time (project1 had to retrofit it after review pressure; do it as
  part of the initial design here).
- **New metrics** (temporal analogs of project1's leakage suite):
  - `trajectory_IoU` / `temporal_AUC`: attacker given a $k$-frame window predicts
    the mask trajectory; score via mean IoU/AUC across the window.
  - a **temporal-consistency-of-leak** score: does the attacker's predicted mask
    move smoothly frame-to-frame (optical-flow-consistent trajectory) vs. jitter
    independently per frame? A smooth, flow-consistent "leaked" trajectory is
    much stronger evidence of real identity/motion leakage than independent
    per-frame noise that happens to overlap the true mask by chance, this
    distinguishes genuine temporal leakage from spatial leakage merely observed
    at multiple timesteps.
- **Oracle floor / $2\sigma$ threshold protocol**: unchanged from project1
  (\cref{eq:threshold} logic ports directly, just compute floor and threshold
  over the temporal attacker's AUC instead of the per-frame one).

## 4. What's explicitly deferred (not scope for the main paper)

- Instance-level memory unlearning in SAM2/XMem/Track-Anything (design fork B
  above), flag as future work / a short pilot case study at most.
- Multimodal leakage (video-LLM prompted questions like "describe the movement
  near the sidewalk"), interesting for a discussion section, not an experiment,
  given compute cost of video-LLM fine-tuning is far outside current envelope.
- "Volumetric leakage" (3D/medical), separate project if pursued at all.

## 5. Immediate next steps

1. Download VSPW (43 GB, ungated) and verify the "person" class index + per-frame
   mask format matches the `ADE20KSeg`-style loader conventions in project1's
   `seg/data.py`, so the dominant-neighbour relabelling code ports with minimal
   change.
2. Stand up a `seg/` package here mirroring project1's (`data.py`, `model.py`,
   `engine.py`) but video-aware: dataset returns a clip (list of frames + masks),
   not a single image.
3. Train the original (no forgetting) DeepLabV3-R50 + temporal module baseline on
   VSPW person class; establish the retrain-from-scratch oracle exactly as in
   project1's `train_seg.py --mode oracle`.
4. Build the temporal attacker zoo (CNN/ConvGRU/3D-conv window attacker) and
   confirm a floor exists (oracle's temporal AUC > chance, mirroring project1's
   finding that even the oracle leaks somewhat via context).
5. Only then port ADVR itself to the temporal setting (retain-replay + adversarial
   discriminator, now windowed).

Do not skip step 4 to jump to method-building, project1's single biggest source
of rigor was establishing the floor *before* claiming any method beats it, and the
AAAI reviewer specifically pressure-tested that floor's stability (see
`project1/paper-AAAI/AAAI_ADVR/sec/8_supplementary.tex`, S.7). Establish the video
floor's stability under at least 2-3 relabelling/attacker variations before
building ADVR-video on top of it, rather than retrofitting that check under
deadline pressure like project1 had to.

## 6. Code status (as of 2026-07-21), apparatus complete, validated

The full `seg/` package + driver scripts are written and smoke-tested end-to-end
(CPU, synthetic + one real VSPW clip). NONE of it is placeholder/toy, every
method is a faithful port of project1's audited implementation or a real new
implementation, shape-checked and run for >=1 step. What exists:

- `seg/data.py`: `VSPWSeg` (per-frame), `VSPWUnlearn` (img,orig,merged),
  `VSPWClip` (temporal clips, clip-consistent aug, cached index). Ignore handling
  fixed for VSPW's dual 0/255 ignore convention (real bug caught via smoke test).
- `seg/methods.py` + `seg/baselines/` (8 published baselines, copied verbatim from
  project1, dataset-agnostic, imports resolve to project3's seg.data).
- `seg/attack.py`: per-frame zoo (cnn/unet/vit, ported) + **temporal attackers
  (conv3d TemporalSilhouetteNet, bidirectional ConvGRUAttacker)** + clip field
  extraction + per-frame and temporal train/eval (temporal returns trajectory
  IoU/AUC + traj_consistency).
- `seg/candidates.py`: `adv_replay` (per-frame ADVR, ported) + **`adv_temporal`
  (ADVR-T, the method)**.
- `seg/audit.py`: per-frame `leakage_metrics` + `relearn_audit` (ported) +
  `trajectory_leakage` (temporal).
- scripts: `train_seg.py`, `run_baseline.sh` (refs), `run_baselines.py` (8
  baselines), `test_candidate.py` (ADVR/ADVR-T), `run_benchmark.py` (utility +
  per-frame leak table), `run_temporal_attack.py` (the headline temporal-vs-frame
  leakage table), `run_all.sh` (chains stages, skip-if-done).

### The ADVR-T objective (the "slight novelty tuned for video")

Per training step over a clip $(x_{1:T}, m_{1:T})$, with a frozen teacher
$f_{\theta^*}$, a **temporal discriminator** $D^T_\phi$ over the clip's field
sequence, and a retain-replay stream:

- **discriminator step**: $\phi \leftarrow \phi - \eta_D \nabla_\phi\,
  \mathrm{BCE}(D^T_\phi(\{O_t\}_{1:T}),\, m_{1:T})$, $D^T$ reconstructs the mask
  *trajectory* (this is what a per-frame ADVR's $D$ cannot do).
- **model step**: descend on
  $\;\mathcal{L}_{\text{blend}} + \lambda_r \mathcal{L}_{\text{KL-retain}}
  + \lambda_{\text{adv}}\,\mathrm{BCE}(D^T_\phi(\{O_t\}),\, \mathbf{0})[\text{forget}]
  + \lambda_{tc}\,\mathcal{L}_{tc}$, plus a hard-CE retain-replay step.
- $\mathcal{L}_{tc}$ (temporal-consistency, `_temporal_consistency`): the forgotten
  region's frame-to-frame output change must not exceed its retain-context's, the
  hole moves with the background, not as an independent person trajectory.

ADVR makes each frame safe ($I(M_t;O_t)\!\to\!0$); **ADVR-T makes the sequence
safe** ($I(M_{1:T};O_{1:T})\!\to\!0$). The paper's headline is the gap: baselines
(and per-frame ADVR) can be per-frame-safe yet trajectory-leaky; only ADVR-T
closes both, measured by `run_temporal_attack.py` (frame vs temporal AUC vs the
oracle floor on each surface).

### Compute reality (measured, not estimated)

Full-res VSPW JPEG decode is the bottleneck: DeepLabV3-R50 trains at ~0.32 s/iter
(bs6, 512, 32-core, compute-bound not I/O-bound, 24 workers didn't help). 4
epochs ~= 132k iters ~= 12 h/model, chosen because 4 epochs already exceeds
project1's ADE20K iteration budget (VSPW has ~13x more person frames/epoch), so
this is a principled setting, not a shortcut. bs8 OOMs the 16 GB card -> bs6.
References (original+oracle) run via `run_baseline.sh` under `setsid` (survives
session teardown; plain nohup+disown did NOT, the first launch was killed at
~1 min). On the 5090 (32 GB) bs can go higher and this shrinks.

### Training crash + fix (2026-07-21, second restart)

`original` crashed at the END of epoch 1 (training itself completed fine, loss
1.841) inside the post-epoch validation `seg_metrics` call: `RuntimeError: shape
'[124,124]' is invalid for input of size 31315`. Root cause: a full-dataset scan
found **17 val mask files with stray raw pixel value 253**, not one of VSPW's 124
documented classes, real annotation corruption (e.g.
`data/353_-5sqmCF8OFU/mask/00000404.png`), not a code bug. My original small
sample (150/198k frames) missed it. It only surfaced in validation, not training,
because val uses uncropped full frames while train uses random 512-crops that
happened to avoid the corrupted region during epoch 1, this could still have hit
training in a later epoch, so it needed a real fix, not a "just re-run" shrug.

**Fix**: `seg/data.py` now treats ANY raw value outside `{0} ∪ {1..124} ∪ {255}`
as ignore (`label[(raw==0)|(raw>124)] = IGNORE`), not just 0/255 specifically,
defensive against any other undocumented stray ID, verified against the actual
corrupted file. Fixed in both `VSPWSeg._load` and `VSPWClip.__getitem__` (was
duplicated). Verified the exact crashing frame now processes cleanly through
`seg_metrics`. Training restarted under `setsid` (again, required every time,
plain background launch does not survive session/tool teardown).

A background full-scan of the train split (2806 videos, slower) was still running
non-blocking at last check, informational only, since the fix already handles
any value it might find without further code changes.

### Immediate next actions when refs finish

1. `bash scripts/run_all.sh` (baselines -> ADVR -> ADVR-T -> benchmark -> temporal
   attack). It's resumable.
2. FIRST real result to look at: does the oracle's TEMPORAL AUC exceed its
   PER-FRAME AUC? If yes, the temporal floor is genuinely higher -> temporal
   context leaks more than spatial (expected, and good for the thesis). If the
   temporal attacker doesn't beat the per-frame attacker on the *original* model
   either, the temporal-attacker capacity/clip_len needs tuning before any method
   claim, do that on the original model first (it's the easiest to leak from).

## 7. Machine reboot mid-pipeline (2026-07-23), third `setsid` survival gap found

`run_all.sh` was ~22h into the baselines stage (6/8 done: FT, NegGrad, NegGrad+,
RL, SalUn, SSD all checkpointed; SCRUB ~350min in, healthy) when **the machine
itself rebooted** at 15:59:48 (`uptime -s` / `last reboot` confirmed), killing
everything including `setsid`-launched processes. `setsid` survives *session/tool
teardown* (the earlier failure mode) but obviously not a full machine restart,
different failure class, not a regression of the earlier fix.

**What survived**: all 6 completed baseline checkpoints (each saved to disk on its
own completion, so no data lost there) + both reference models. **What was lost**:
SCRUB's in-progress training only (no checkpoint until `run_baselines.py`'s
per-baseline save, so ~350min of that one run had to restart from scratch).

**Resumed cleanly**: `run_baselines.py`'s skip-if-exists check meant relaunching
`run_all.sh` automatically skipped the 6 done baselines and picked up at SCRUB.
This relaunch is also the first to run under the person-frame-scan CACHE fix
(§ code status) since the crashed process had the pre-fix code loaded in memory,
so this run pays the ~8min scan once and writes the cache for ADVR/ADVR-T/
benchmark/temporal-attack to reuse.

**Lesson for future long runs on this machine**: no defense against a full reboot
exists at the process-supervision level (would need e.g. a systemd unit or cron
`@reboot` re-launcher to auto-resume unattended). Given the pipeline is already
resumable via skip-if-exists checkpointing, the practical mitigation is just:
after any gap in monitoring, always verify with `uptime -s` whether the machine
restarted before assuming a process merely crashed.

## 8. Protocol audit (2026-07-26), the T-sweep was confounded; thesis restated

A code read of the measurement path, prompted by the suspiciously clean
"ADVR-T strengthens with T" curve, found three defects. The first two
invalidate `results/seg/sweep_clip_len.log` and every `temporal_attack_T*.json`
built from it. **Those numbers must not go in the paper.**

### 8.1 The tell

The retrain-from-scratch oracle floor moved across the sweep:

| T | 2 | 3 | 5 | 8 | 16 |
|---|---|---|---|---|----|
| floor IoU | 17.7 | 15.2 | 5.8 | 0.0 | 0.7 |

The floor is the irreducible context-prior leakage of a model that never saw
"person". It has no mechanism by which to depend on the attacker's window
length. That it fell to zero is proof the *measurement* changed, not the
phenomenon, and every method's curve inherited the same downward drift, which
is exactly what was mistaken for "ADVR-T strengthens with T".

### 8.2 Defect 1, evaluation content varied with T (severe)

`VSPWClip` tiles with `stride=clip_len`, so clip count collapses with T; the
loader then took the *first* `N_VAL` clips, which are ordered by video. Measured
on the real cache:

| T | clips available | clips scored | distinct val videos |
|---|---|---|---|
| 2 | 8311 | 1500 | **36** |
| 3 | 5557 | 1500 | 50 |
| 5 | 3335 | 1500 | 77 |
| 8 | 2008 | 1500 | 166 |
| 16 | 896 | 896 (all) | **238** |

T=2 was scored on 36 videos, T=16 on 238. A "leakage vs T" curve built this way
is confounded with dataset composition. (Person-density per clip was checked as
a possible third confound and ruled out: 99.9% → 98.6%.)

**Fix:** `seg/data.VSPWClipAnchored`, anchors drawn once from the split alone,
every T windows the *same* anchors. Verified: 1200 clips / 231 videos /
byte-identical anchor list at T ∈ {2,3,5,8,16}.

### 8.3 Defect 2, attacker budget shrank as T grew (severe)

`MAX_ITERS=500` was fixed while `BS_CLIP` was scaled 8→4→2→1 for memory, so
clips seen went 4000 → 2000 → 1000 → **500**, with batch-size-1 gradients at
T=16. The attacker gets monotonically weaker in T, so measured leakage falls in
T for free. A leakage result is a claim about *absent* information; it cannot
rest on an attacker that was starved.

**Fix:** fixed optimizer steps *and* fixed effective batch in clips at every T
via gradient accumulation (`train_temporal_attacker(accum_steps=...)`), plus
`D.train_losses` so a converged-but-empty attacker is distinguishable from one
that never trained.

### 8.4 Defect 3, IoU@0.5 is a calibration artifact, not a privacy measure

ADVR-T at T=16 scored **IoU 0.0 with AUC 61.2**, above chance. The attacker
ranked person pixels correctly; its sigmoid simply never crossed 0.5. Reported
as "0.0", that reads as "no leakage" when the truth is "leakage present,
attacker uncalibrated". Reproduced in a controlled synthetic case: an attacker
at **AUC 94.7** scores IoU@0.5 = 2.3 while its best-threshold IoU is 37.7.
`traj_consistency` had the matching bug, `max(active,1)` returned a
perfect-looking `100.0` for an all-negative prediction (fired twice: oracle at
T=8, ADVR-T at T=16).

**Fix:** `seg/metrics.py`, exact 512-bin score histograms over every pixel
(no subsampling) giving AP (primary), AUC, IoU@0.5, IoU_best, pos_rate;
`traj_consistency` returns NaN when degenerate. Validated against sklearn.

### 8.5 The confound that survives all of the above: utility

From `benchmark.json`: retain_mIoU is oracle **30.8**, ADVR 24.2, ADVR-T
**22.5**. ADVR-T's output field is globally degraded relative to the oracle, so
part of its "privacy" is simply that there is less of everything for an attacker
to read, which is also why it scored *below* a floor that is supposed to be
irreducible. **Leakage is only meaningful at matched utility.** Any headline
comparison must either (a) hold retain_mIoU fixed by tuning `lam_r`/replay, or
(b) report the privacy–utility Pareto frontier. Leakage alone rewards damage,
and a trivially blank model wins it.

### 8.6 Restated thesis: separate redundancy from synergy

"Temporal attacker beats per-frame attacker" was the wrong claim, because two
different mechanisms produce it and only one is novel:

1. **Redundancy / denoising.** Masks are temporally correlated, so a temporal
   attacker averages T noisy per-frame estimates and cuts variance. Order-free,
   grows with T for any per-frame-leaky model, and completely unsurprising.
2. **Synergy.** Information carried by the trajectory itself, motion signature,
   differential cues, present in no single frame. Order-dependent. *This* is
   temporal silhouette leakage.

Time-shuffling a clip (same permutation on fields and masks) destroys 2 and
leaves 1 untouched. So the three nested attacker capabilities are:

```
AP(per-frame)  --Δ_redundancy-->  AP(time-shuffled temporal)  --Δ_trajectory-->  AP(ordered temporal)
```

and the paper's claim lives entirely in **Δ_trajectory**. If Δ_trajectory ≈ 0,
temporal silhouette leakage reduces to denoising and the thesis needs
rewriting, this is a real risk the old protocol could not have detected, and
the 25-step smoke test was not reassuring (`original` showed Δ ≈ −0.5, though
at a budget far too small to mean anything). `scripts/run_temporal_protocol.py`
measures all three levels.

### 8.7 Method: ADVR-TX (`seg/candidates.adv_temporal_x`)

Two upgrades, both aimed at defects in ADVR-T's objective rather than at the
numbers:

* **Local-ring `L_tc`.** ADVR-T compares the hole's frame-to-frame change to the
  *global mean* retain change, one scalar per frame-pair. But field-change is
  spatially heterogeneous (boundaries move, interiors do not), so a hole in a
  busy region gets a lenient budget to hide a trajectory in, while a hole in a
  static region is over-penalised and pays utility for nothing. The reference is
  now the mean change over a dilated *ring around that hole*: "move like the
  scene right around you."
* **Temporal-exchangeability adversary.** Magnitude is a proxy, a model can
  respect a change budget and still leave a readable trajectory by moving
  smoothly. `OrderDiscriminator` sees the field sequence masked to a dilated
  neighbourhood of the hole and must decide *true order vs permuted*; the model
  is trained to make that impossible. A sequence with no order information has
  no trajectory, stated directly rather than via a surrogate. It also closes the
  loop with §8.6: the objective drives down precisely the Δ_trajectory the
  protocol reports, so the method is not graded on a quantity it never optimised.

Validated on CPU: local `L_tc` ≈ 0 for a hole matching its surroundings vs
0.0138 for one moving independently; full ADVR-TX step updates all params,
stays finite, and handles `lam_ord=0` and T=1 without crashing. **Not yet
trained**, needs GPU time after the protocol sweep.
