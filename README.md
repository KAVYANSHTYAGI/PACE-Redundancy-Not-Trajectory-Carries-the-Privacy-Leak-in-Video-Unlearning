# PACE: Redundancy, Not Trajectory, Carries the Privacy Leak in Video Unlearning

Code and experiment logs for **"Beyond the Frame: Redundancy,
Not Trajectory, Carries the Privacy Leak in Video Unlearning"** (WACV 2027).

Project page: https://kavyanshtyagi.github.io/PACE-Redundancy-Not-Trajectory-Carries-the-Privacy-Leak-in-Video-Unlearning/

## TL;DR

Class-unlearning methods for video segmentation are usually audited one frame
at a time. We show this understates leakage badly: an adversary that simply
**averages** a model's output field over a ~2-second clip recovers the
forgotten class from *every* published unlearning baseline we test, including
methods whose individual frames are visually indistinguishable from a
retrained oracle. A time-shuffled control shows the recoverable signal comes
from **cross-frame redundancy**, not object trajectory (shuffling frame order
does not destroy it). We introduce **PACE**, which suppresses this directly
with an exactly permutation-invariant pooled adversary plus a two-sided
temporal-consistency objective, and validate it on two datasets (VSPW,
Virtual KITTI 2) and two forget classes (`person`, `car`).

**Headline numbers** (pooled attacker AP after unlearning, lower is safer;
retain mIoU, higher is better). The oracle is a model retrained without the
forgotten class.

| Method | VSPW person (mIoU / AP) | VSPW car (mIoU / AP) | VKITTI2 car (mIoU / AP) |
|---|---|---|---|
| Oracle (retrain) | 30.8 / 30.82 | 31.6 / 18.10 | 45.0 / 50.71 |
| Original | 33.5 / 93.82 | 31.9 / 80.45 | 50.1 / 88.67 |
| SalUn | 15.1 / 85.68 | 4.5 / 62.85 | 32.2 / 92.42 |
| SCRUB | 31.2 / 88.60 | 27.5 / 54.41 | 44.7 / 95.66 |
| PACE-T | 22.5 / 21.10 | 20.2 / 13.93 | 21.1 / 17.12 |
| **PACE** | **21.2 / 18.79** | **19.8 / 12.10** | **7.6 / 9.11** |

SSD, NegGrad and NegGrad+ reach low AP on some classes by collapsing retain
mIoU (3.6 or less on car), so low AP alone is not a useful result. The full
table with all baselines is on the project page and in the paper.

Full numbers, ablations, and every negative/inconclusive result along the way
are in [`EXPERIMENTS.md`](EXPERIMENTS.md) (the running lab notebook, treat it
as the primary source of truth over any summary here) and
[`POSITIONING.md`](POSITIONING.md) (research framing/narrative).

## See it leak, in motion

Static frame grids (Figure 4 in the paper) only show one instant. Below is
the *same 16-frame validation clip*, rendered in full, for six methods side
by side, generated directly from the checkpoints and evaluation code in this
repo (`scripts/make_qual_video.py`), not staged illustrations.

Mean recovered probability for the forgotten class on this clip: `original`
0.547, `RL` 0.524, `SCRUB` 0.494, `SalUn` 0.551, `PACE-T` 0.166, **`PACE`
0.112**. Every per-frame-safe baseline still lights up once pooled; PACE is
the one that doesn't.

The clips play on the [project page](https://kavyanshtyagi.github.io/PACE-Redundancy-Not-Trajectory-Carries-the-Privacy-Leak-in-Video-Unlearning/), one strip with every method's own prediction and one with what the pooled attacker recovers. Its source is in [`docs/`](docs/index.html).

## Repository layout

```
seg/                  Core library
  data.py              VSPW datasets: per-frame, unlearning pairs, clips (VSPWClip / VSPWClipAnchored)
  vkitti_data.py        Same, for Virtual KITTI 2
  model.py             DeepLabV3-ResNet50 factory (get_seg_model, seg_forward)
  methods.py           Pixel-level unlearning baselines (FT, NegGrad, ...)
  baselines/           Baselines ported from the companion image-unlearning project (project1):
                        delete, finetune, neggrad, neggrad_plus, random_label, salun, scrub, ssd
  candidates.py        The paper's method line: ADVR (per-frame), ADVR-T (temporal), ADVR-P (pooled)
                        (named PACE-F, PACE-T and PACE in the paper)
  attack.py            All attacker architectures (conv3d, convgru, mamba, DeepSets) + train/eval loops
  mamba.py             Selective-state-space (Mamba) temporal-mixing block used by the strongest attacker
  metrics.py           Threshold-free scoring (AP / AUC / IoU_best), trajectory-consistency metric
  audit.py             leakage_metrics, relearn_audit, trajectory_leakage, the auditing entry points
  engine.py, vkitti_engine.py   Train/eval loops

scripts/               Every experiment driver. See EXPERIMENTS.md §7 for which
                        scripts are CURRENT vs SUPERSEDED, this is a living
                        research repo, not every script here fed the final paper.
  download_vspw.sh      Fetches and lays out the VSPW dataset
  run_baseline.sh        Trains the two reference models (original, oracle)
  run_all.sh             Full downstream pipeline: baselines -> ADVR -> ADVR-T -> benchmark -> temporal attack
  run_advrp.sh           Trains PACE (ADVR-P)
  run_temporal_protocol.py   The main pooled-attacker evaluation protocol
  train_seg.py, test_candidate.py, run_baselines.py   Lower-level training entry points
  make_*.py              Figure-generation scripts for the paper
  make_page_assets.py    Cuts the strip videos into the sprite sheets used by docs/index.html

results/seg/            Every JSON result and full stdout log this project produced,
                        kept as-is for reproducibility auditing (see EXPERIMENTS.md
                        for which files are the ones actually cited in the paper,
                        several are superseded protocol runs kept only for the
                        historical record).

docs/                   GitHub Pages project site: index.html (self-contained page with the
                        interactive viewer, diagrams and results), assets/videos (the two strip
                        videos) and assets/frames (per-method sprite sheets cut from them by
                        scripts/make_page_assets.py).

checkpoints/            NOT included in this repository (see "Trained weights" below).
data/                   NOT included in this repository (see "Datasets" below).
```

## Setup

```bash
git clone https://github.com/KAVYANSHTYAGI/PACE-Redundancy-Not-Trajectory-Carries-the-Privacy-Leak-in-Video-Unlearning.git
cd PACE-Redundancy-Not-Trajectory-Carries-the-Privacy-Leak-in-Video-Unlearning
python -m venv .venv && source .venv/bin/activate   # or use your existing conda env
pip install -r requirements.txt
```

Developed and run on Python 3.12, PyTorch 2.9.0 (CUDA 12.8), a single 16 GB
GPU (RTX 5080). Nothing here assumes multi-GPU; scripts are written for a
single-card, resumable, long-running-job workflow (see "Orchestration
conventions" below).

## Datasets

Not included in this repository (VSPW alone is ~44 GB; distributing third-party
research datasets in a code repo is both impractical and outside their
license terms). Both are freely available for research use directly from
their official sources:

- **VSPW** (Video Scene Parsing in the Wild, 124 classes, 3536 videos):
  `scripts/download_vspw.sh` fetches and lays it out into `data/VSPW/` in the
  layout `seg/data.py` expects. See the [official VSPW
  repository](https://github.com/sssdddwww2/vspw_dataset_download) for the
  underlying source and terms.
- **Virtual KITTI 2**: download from the [official Virtual KITTI 2
  page](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)
  and place under `data/VKITTI2/`; `seg/vkitti_data.py` documents the expected
  subdirectory structure at the top of the file.

Class-index conventions (raw label → training index, ignore-label handling,
the one real annotation-corruption bug we found and how it's handled) are
documented in `EXPERIMENTS.md` §2, read this before touching `seg/data.py`,
it is not obvious from the code alone.

## Trained weights

Checkpoints are not included in this repository (47 checkpoints, ~7.9 GB
total, over GitHub's practical size limits, and every one of them is exactly
reproducible from the scripts here given the datasets above). To regenerate:

1. `bash scripts/run_baseline.sh`: trains the two reference models
   (`original`: fully trained, nothing forgotten; `oracle`: retrained from
   scratch with the forget class excluded, the two anchors every comparison
   in the paper is measured against).
2. `bash scripts/run_all.sh`: trains all 8 baselines plus ADVR/ADVR-T, then
   runs the benchmark and temporal-attack evaluation.
3. `bash scripts/run_advrp.sh`: trains PACE (ADVR-P), the paper's method.
4. For the `car` class and Virtual KITTI 2 variants, see
   `scripts/run_car_pipeline.sh` and `scripts/run_vkitti_pipeline.sh`
   respectively.

Expect roughly 1 to 14 hours per model on a single 16 GB GPU depending on method
(baselines are cheap; ADVR-T/PACE's adversarial training is the expensive
end), see the `elapsed=` timestamps in `results/seg/*.log` for exact
historical wall-clock times on this hardware.

## Reproducing the headline result

```bash
export PYTHONPATH=.
# 1. reference models
bash scripts/run_baseline.sh
# 2. baselines + PACE's predecessors (ADVR, ADVR-T)
bash scripts/run_all.sh
# 3. PACE itself
bash scripts/run_advrp.sh
# 4. the pooled-attacker protocol that produces the paper's headline table
python scripts/run_temporal_protocol.py
```

Each stage is resumable, every script checks for its own output checkpoint/
JSON before retraining, so a killed/restarted run picks up where it left off
rather than redoing finished work.

## The core idea, in code terms

The paper's central methodological point is that **per-frame auditing is not
enough**, you must audit a *pooled* view of multiple frames, because that is
what an adversary would actually do. Concretely:

- `seg/metrics.py` / `seg/audit.py` implement the ordered-pool audit
  (`AP_ordered`) and, critically, a **time-shuffled control**
  (`AP_shuffled`), if shuffling frame order before pooling doesn't reduce
  leakage, the signal isn't coming from motion/trajectory, it's coming from
  redundant per-frame content. `synergy = AP_ordered - AP_shuffled` and
  `Δ_red = AP_shuffled - AP_frame` are the two diagnostic quantities used
  throughout the paper and `EXPERIMENTS.md` to tell these apart.
- `seg/candidates.py` implements the three methods along the paper's method
  line: `ADVR` (per-frame adversarial suppression only), `ADVR-T` (adds a
  temporal-consistency term), and `ADVR-P`/**PACE** (adds an exactly
  permutation-invariant *pooled* adversary `D_P` on top of ADVR-T, the full
  method).
- `seg/attack.py` implements every attacker architecture used to stress-test
  the above, escalating in expressiveness: raw-logit pooling
  (`PoolDiscriminator`), 3D-conv / ConvGRU / Mamba temporal mixers, and a
  `DeepSetsAttacker` (learned per-frame embedding, mean-pooled, decoded) that
  is strictly more expressive than raw-logit pooling while remaining exactly
  permutation-invariant, a claim verified numerically at the bottom of the
  file, not just asserted.

## Orchestration conventions (if you resume a long run)

Every `run_*.sh` script follows the same pattern used throughout this
project for sequencing long single-GPU jobs without contention:

```bash
setsid nohup <command> > results/seg/<name>.log 2>&1 < /dev/null &
disown
```

with wrapper scripts (`*_wrapper.log` in `results/seg/`) waiting on a
specific PID before launching the next stage, to avoid two training jobs
racing for the same 16 GB of VRAM. If you adapt these scripts, wait on the
**wrapper script's own PID**, not a sub-process PID inside it, the one real
orchestration bug hit during this project (documented in `EXPERIMENTS.md`)
came from exactly that mistake.

## Citation

```bibtex
@inproceedings{tyagi2027beyond,
  title     = {Beyond the Frame: Redundancy, Not Trajectory, Carries the Privacy Leak in Video Unlearning},
  author    = {Tyagi, Kavyansh and Russo, Paolo},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## License

Code in this repository (everything under `seg/`, `scripts/`) is released
under the MIT License, see [`LICENSE`](LICENSE).
