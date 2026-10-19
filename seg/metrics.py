"""Threshold-free leakage metrics via exact score histograms.

WHY THIS EXISTS. The original protocol scored an attacker with IoU at a
hardcoded 0.5 threshold plus an AUC estimated from a 20k-pixel random subsample.
Both are unsafe for the regime this paper lives in (rare positives, attackers
driven toward the noise floor):

  * IoU@0.5 collapses to exactly 0.0 as soon as the attacker is under-confident,
    even when its *ranking* is clearly informative. Observed in the T-sweep:
    ADVR-T at T=16 scored IoU 0.0 while AUC was 61.2 -- i.e. well above chance.
    Reporting "IoU 0.0" there reads as "no leakage" when the truth is "the
    attacker found signal but never crossed 0.5". That is a calibration
    artifact, not a privacy result.
  * A leakage claim is a claim about the *absence* of recoverable information,
    so it must not rest on a metric that a mis-calibrated or under-trained
    attacker can trivially satisfy.

So we score the attacker with:
  AP      -- average precision, the right summary for rare-positive masks and
             invariant to the attacker's output calibration.
  AUC     -- rank-based, threshold-free.
  IoU@0.5 -- kept only for continuity with the earlier numbers.
  IoU*    -- IoU at its BEST threshold (the attacker's achievable reconstruction
             if it were calibrated). IoU* is the honest upper bound on what this
             attacker recovers; IoU@0.5 is a lower bound confounded by calibration.
  pos_rate-- fraction of pixels predicted positive at 0.5, purely diagnostic: it
             tells "attacker collapsed to all-negative" (pos_rate~0) apart from
             "attacker genuinely localised nothing".

All of these come from a 512-bin histogram of scores accumulated over EVERY
pixel -- exact, no subsampling, and cheap (one bincount per batch on GPU).
"""
import numpy as np
import torch

_NAN = float("nan")


class ScoreHist:
    """Accumulates per-pixel attacker scores into positive/negative histograms.

    Exact over all pixels seen (no subsampling), so every derived metric below
    is a population quantity, not an estimate with sampling noise.
    """

    def __init__(self, bins=512, device="cpu"):
        self.bins = bins
        self.pos = torch.zeros(bins, dtype=torch.float64, device=device)
        self.neg = torch.zeros(bins, dtype=torch.float64, device=device)

    @torch.no_grad()
    def add(self, prob, gt):
        """prob: sigmoid scores in [0,1], any shape. gt: same shape, {0,1}."""
        prob = prob.detach().flatten()
        g = gt.detach().flatten() > 0.5
        finite = torch.isfinite(prob)
        if not finite.all():
            # A diverged attacker (NaN logits) shouldn't crash the whole eval
            # run -- drop the non-finite pixels and keep scoring the rest.
            n_bad = int((~finite).sum())
            print(f"[ScoreHist] dropping {n_bad} non-finite score(s)", flush=True)
            prob, g = prob[finite], g[finite]
        prob = prob.clamp(0.0, 1.0)
        idx = (prob * (self.bins - 1)).round().long()
        pos = torch.bincount(idx[g], minlength=self.bins)
        neg = torch.bincount(idx[~g], minlength=self.bins)
        self.pos += pos.to(self.pos.device, self.pos.dtype)
        self.neg += neg.to(self.neg.device, self.neg.dtype)

    def metrics(self, prefix=""):
        pos = self.pos.cpu().numpy().astype(np.float64)
        neg = self.neg.cpu().numpy().astype(np.float64)
        P, N = pos.sum(), neg.sum()
        k = lambda s: f"{prefix}{s}"
        if P < 1 or N < 1:
            return {k("AP"): _NAN, k("AUC"): _NAN, k("IoU"): _NAN,
                    k("IoU_best"): _NAN, k("thr_best"): _NAN, k("pos_rate"): _NAN}

        # Threshold at bin i == "predict positive iff score bin >= i".
        tp = np.cumsum(pos[::-1])[::-1]          # positives at or above bin i
        fp = np.cumsum(neg[::-1])[::-1]          # negatives at or above bin i
        fn = P - tp
        iou = tp / np.maximum(tp + fp + fn, 1e-12)
        prec = tp / np.maximum(tp + fp, 1e-12)
        rec = tp / P

        # AP: sum of precision weighted by the recall it gains, sweeping the
        # threshold down (rec is non-increasing in bin index).
        rec_next = np.concatenate([rec[1:], [0.0]])
        ap = float((prec * (rec - rec_next)).sum())

        # AUC (Mann-Whitney with ties at 0.5), exact from the histogram.
        neg_below = np.concatenate([[0.0], np.cumsum(neg)[:-1]])
        auc = float((pos * (neg_below + 0.5 * neg)).sum() / (P * N))

        i50 = int(round(0.5 * (self.bins - 1)))
        ibest = int(np.argmax(iou))
        return {
            k("AP"): 100 * ap,
            k("AUC"): 100 * auc,
            k("IoU"): 100 * float(iou[i50]),
            k("IoU_best"): 100 * float(iou[ibest]),
            k("thr_best"): ibest / (self.bins - 1),
            k("pos_rate"): 100 * float((tp[i50] + fp[i50]) / (P + N)),
        }


class TrajConsistency:
    """Frame-to-frame stability of the attacker's PREDICTED mask.

    Fixed vs the original: the old version divided by max(active, 1), so an
    attacker predicting nothing anywhere returned a perfect-looking 100.0. That
    fired twice in the sweep (oracle at T=8, ADVR-T at T=16) and is meaningless
    -- there is no trajectory to be consistent about. Now returns NaN below
    `min_active` predicted pixels, so a degenerate attacker is reported as
    "undefined", never as "perfectly consistent".
    """

    def __init__(self, min_active=1000):
        self.flip = 0
        self.active = 0
        self.min_active = min_active

    @torch.no_grad()
    def add(self, pred):
        """pred: bool [N,T,h,w] thresholded prediction; needs T>1."""
        if pred.shape[1] < 2:
            return
        change = pred[:, 1:] != pred[:, :-1]
        active = pred[:, 1:] | pred[:, :-1]
        self.flip += int(change.sum().item())
        self.active += int(active.sum().item())

    def value(self):
        if self.active < self.min_active:
            return _NAN
        return 100.0 * (1.0 - self.flip / self.active)
