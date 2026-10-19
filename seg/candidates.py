"""ADVR and its temporal extension ADVR-T (the project3 method).

ADVR (ported from project1): min over the model of
    blend(person->dominant-neighbour) + lam_r * retain-KL-anchor
    + lam_adv * fool(per-frame silhouette D) + retain-replay,
which drives the PER-FRAME spatial leakage I(M_t; O_t) -> 0.

ADVR-T (new, the slight novelty tuned for video): identical, except the
adversary is a TEMPORAL discriminator D^T that reconstructs the person-mask
*trajectory* from a window of output fields (seg/attack.TemporalSilhouetteNet).
Fooling D^T forces the model to remove structure that is only recoverable
across frames -- the trajectory -- which a per-frame D (and hence per-frame
ADVR) never penalises. An optional temporal-consistency regulariser (lam_tc)
additionally requires the forgotten region's output to change over time like its
surrounding retain context (move with the background), not like an independently
moving person.

POSITIONING.md's temporal HYP in one objective: ADVR makes each frame safe;
ADVR-T makes the *sequence* safe.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from seg.data import IGNORE, NUM_CLASSES, PERSON
from seg.methods import _clone, _masked_ce, _masks
from seg.model import seg_forward


def _frozen(model, device):
    t = _clone(model).to(device).eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def _freeze_bn(model):
    for mod in model.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.eval()


def _masked_kl(z_s, z_t, mask, T=1.0):
    if mask.sum() == 0:
        return z_s.sum() * 0.0
    ls = F.log_softmax(z_s / T, 1)
    pt = F.softmax(z_t / T, 1)
    kld = F.kl_div(ls, pt, reduction="none").sum(1)        # [N,H,W]
    return (kld * mask).sum() / mask.sum().clamp(min=1) * (T * T)


# --------------------------------------------------------------------------- #
# ADVR (per-frame) -- ported from project1 adv_full (lam_c=0 default here, the
# feature-collapse term is dropped; it was a minor ablation there and the video
# story is about the temporal axis, not the feature surface).
# --------------------------------------------------------------------------- #
def adv_replay(model, loader, device="cuda", epochs=6, lr=1e-3, lam_r=1.0,
               lam_adv=1.0, lr_d=1e-3, out_size=128, replay_loader=None,
               freeze_bn=False, **_):
    """Per-frame ADVR. `loader` yields (img, orig, merged) single frames."""
    from seg.attack import SilhouetteNet, person_mask_low

    m = _clone(model).to(device).train()
    if freeze_bn:
        _freeze_bn(m)
    teacher = _frozen(model, device)
    D = SilhouetteNet().to(device).train()
    optG = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    optD = torch.optim.Adam(D.parameters(), lr=lr_d)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)
    replay_it = iter(replay_loader) if replay_loader is not None else None

    def field(z):
        return F.interpolate(F.softmax(z, 1), size=(out_size, out_size),
                             mode="bilinear", align_corners=False)

    for _e in range(epochs):
        if freeze_bn:
            _freeze_bn(m)
        for img, orig, merged in loader:
            img, orig, merged = img.to(device), orig.to(device), merged.to(device)
            forget, retain = _masks(orig)
            Mlow = person_mask_low(orig, out_size)

            with torch.no_grad():
                p_det = field(seg_forward(m, img))
            optD.zero_grad(); bce(D(p_det), Mlow).backward(); optD.step()

            z = seg_forward(m, img)
            with torch.no_grad():
                z0 = seg_forward(teacher, img)
            loss = _masked_ce(z, merged, forget) + lam_r * _masked_kl(z, z0, retain)
            dpred = D(field(z)); fmask = Mlow > 0.5
            if fmask.any():
                loss = loss + lam_adv * bce(dpred[fmask], torch.zeros_like(dpred[fmask]))
            optG.zero_grad(); loss.backward(); optG.step()

            if replay_it is not None:
                try:
                    rimg, rlab = next(replay_it)
                except StopIteration:
                    replay_it = iter(replay_loader); rimg, rlab = next(replay_it)
                rimg, rlab = rimg.to(device), rlab.to(device)
                optG.zero_grad(); ce(seg_forward(m, rimg), rlab).backward(); optG.step()
    return m


# --------------------------------------------------------------------------- #
# ADVR-T (temporal) -- THE method. `loader` yields VSPWClip batches
# (imgs[N,T,3,H,W], origs[N,T,H,W], mergeds[N,T,H,W]).
# --------------------------------------------------------------------------- #
def _flat(x):
    """[N,T,...] -> [N*T,...]."""
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def adv_temporal(model, loader, device="cuda", epochs=6, lr=1e-3, lam_r=1.0,
                 lam_adv=1.0, lam_tc=0.0, lr_d=1e-3, out_size=128,
                 replay_loader=None, freeze_bn=False, temporal_kind="conv3d", **_):
    """ADVR-T: per-frame blend + retain-KL anchor + retain-replay (as ADVR), but
    the adversary is a TEMPORAL discriminator over the clip's field-sequence, and
    an optional temporal-consistency term (lam_tc) ties the forgotten region's
    motion to its context.

    lam_tc>0 adds L_tc = mean over forget pixels of the squared deviation of their
    frame-to-frame field change from the mean field change of retain pixels in the
    same frame-pair -- i.e. the hole must move with the background, removing any
    independent person-trajectory signal in the output field itself."""
    from seg.attack import build_attacker, clip_person_mask_low

    m = _clone(model).to(device).train()
    if freeze_bn:
        _freeze_bn(m)
    teacher = _frozen(model, device)
    # build_attacker so temporal_kind actually selects the adversary -- the old
    # branch silently fell back to ConvGRU for anything that was not "conv3d",
    # so passing "mamba" would have trained against the wrong discriminator.
    DT = build_attacker(temporal_kind, out_size=out_size).to(device).train()
    optG = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    optD = torch.optim.Adam(DT.parameters(), lr=lr_d)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)
    replay_it = iter(replay_loader) if replay_loader is not None else None

    def clip_field(zf, N, T):
        """zf [N*T,C,H,W] logits -> field [N,T,C,h,w]."""
        p = F.interpolate(F.softmax(zf, 1), size=(out_size, out_size),
                          mode="bilinear", align_corners=False)
        return p.reshape(N, T, *p.shape[1:])

    for _e in range(epochs):
        if freeze_bn:
            _freeze_bn(m)
        for imgs, origs, mergeds in loader:
            imgs = imgs.to(device); origs = origs.to(device); mergeds = mergeds.to(device)
            N, T = imgs.shape[:2]
            imgs_f, origs_f, merged_f = _flat(imgs), _flat(origs), _flat(mergeds)
            forget_f, retain_f = _masks(origs_f)
            Mlow = clip_person_mask_low(origs, out_size)          # [N,T,h,w]

            # ---- D^T step: reconstruct trajectory from detached field-seq ----
            with torch.no_grad():
                O_det = clip_field(seg_forward(m, imgs_f), N, T)
            optD.zero_grad(); bce(DT(O_det), Mlow).backward(); optD.step()

            # ---- G step ----
            zf = seg_forward(m, imgs_f)                            # [N*T,C,H,W]
            with torch.no_grad():
                z0f = seg_forward(teacher, imgs_f)
            loss = _masked_ce(zf, merged_f, forget_f) + lam_r * _masked_kl(zf, z0f, retain_f)

            O = clip_field(zf, N, T)
            dpred = DT(O); fmask = Mlow > 0.5
            if fmask.any():
                loss = loss + lam_adv * bce(dpred[fmask], torch.zeros_like(dpred[fmask]))

            if lam_tc > 0 and T > 1:
                loss = loss + lam_tc * _temporal_consistency(O, Mlow, origs, out_size)

            optG.zero_grad(); loss.backward(); optG.step()

            if replay_it is not None:
                try:
                    rimg, rlab = next(replay_it)
                except StopIteration:
                    replay_it = iter(replay_loader); rimg, rlab = next(replay_it)
                rimg, rlab = rimg.to(device), rlab.to(device)
                optG.zero_grad(); ce(seg_forward(m, rimg), rlab).backward(); optG.step()
    return m


def _temporal_consistency(O, Mlow, origs, out_size):
    """L_tc: the forget region's frame-to-frame field change should match the
    retain context's. O [N,T,C,h,w] (softmax field), Mlow [N,T,h,w] person mask,
    origs [N,T,H,W] original labels (for the retain mask, downsampled)."""
    N, T = O.shape[:2]
    dO = (O[:, 1:] - O[:, :-1]).pow(2).sum(2)                     # [N,T-1,h,w] change magnitude
    fmask = (Mlow[:, 1:] > 0.5) & (Mlow[:, :-1] > 0.5)            # person in both frames
    # retain mask at low res: valid non-person
    rl = F.interpolate(origs.float().reshape(N * T, 1, *origs.shape[2:]),
                       size=(out_size, out_size), mode="nearest").reshape(N, T, out_size, out_size)
    rmask = (rl != PERSON) & (rl != IGNORE)
    rmask_pair = (rmask[:, 1:] & rmask[:, :-1])
    if fmask.sum() == 0 or rmask_pair.sum() == 0:
        return O.sum() * 0.0
    # per (n, frame-pair) mean retain change; forget change should not exceed it
    ref = (dO * rmask_pair).sum(dim=(2, 3)) / rmask_pair.sum(dim=(2, 3)).clamp(min=1)  # [N,T-1]
    ref = ref.unsqueeze(-1).unsqueeze(-1)                        # broadcast
    dev = F.relu(dO - ref)                                       # only penalise EXCESS motion
    return (dev * fmask).sum() / fmask.sum().clamp(min=1)


def _temporal_consistency_local(O, Mlow, origs, out_size, ring_k=9, two_sided=False):
    """Local-ring variant of L_tc -- v2, fixing two mechanistic defects found
    (2026-07-30) by tracing gradients through the v1 local-ring version, after
    v1 was observed to make BOTH utility and privacy worse than the plain
    global-mean L_tc it was meant to improve on (retain_mIoU 22.5->20.9-21.5,
    mamba ordered AP 20.65->22.70-24.81 in the ADVR-TX ablation). Simultaneous
    regression on both axes is the signature of a broken optimization
    incentive, not bad luck -- confirmed by autograd, not assumed:

    DEFECT 1 -- reward hacking via a differentiable reference. v1's `ref` was
    computed from `dO` (the model's own live output) with no stop-gradient, so
    `relu(dO_hole - ref)` has TWO ways to shrink: suppress the hole's motion
    (the intended fix), or INFLATE ref by increasing motion in the ring around
    the hole (a shortcut that damages retain predictions near the boundary
    without touching the hole's real signal at all). Verified directly:
    backprop through v1 puts nonzero gradient on ring pixels outside the hole
    -- confirmed on a synthetic clip. The global-mean version has the same
    channel in principle, but averaged over ~4000+ retain pixels the exploit
    gradient per pixel is ~1.9 MILLION times smaller than the hole's own
    gradient (measured), so it is never worth taking; concentrated over a
    ~9x9 ring (~50-70 pixels) it is cheap and the optimizer found it. Fixed by
    detaching the reference: `dO` computed for `ref` is `.detach()`ed, so
    `ref` is a frozen TARGET the hole must match, not a quantity the model can
    move to game the loss. Verified: ring gradient is now exactly 0.0.

    DEFECT 2 -- a purely local reference can be MORE PERMISSIVE than the
    global one exactly where suppression matters most (e.g. under camera pan,
    the whole local neighbourhood is already moving a lot, handing the hole a
    large "free" motion budget). Fixed by taking `ref = min(local_ref,
    global_ref)`: the hole is never allowed more room than the stable global
    budget, and only gets a STRICTER (smaller) budget when its local context
    is unusually static -- the actually-useful half of the original local-ring
    idea, kept; the exploitable half, removed.

    `two_sided` (2026-07-30, the ADVR-P direction): the original penalty,
    relu(dO - ref), only fires when the hole moves MORE than its reference --
    it is silent whenever the hole is MORE STATIC than context. But the
    mechanistic story behind Delta_redundancy (EXPERIMENTS.md sec 9) is that a
    pooling attacker exploits a FRAME-CONSISTENT bias -- a hole that barely
    changes at all across frames is exactly the shape of thing averaging
    sharpens into a confident estimate, so "too static" is at least as
    dangerous as "too dynamic", and the one-sided version does nothing to
    discourage it. `two_sided=True` replaces relu(dO-ref) with (dO-ref)^2,
    penalising deviation from the reference in EITHER direction -- the
    requirement becomes "match the context's temporal variance", not merely
    "don't exceed it". Uses the same detached, min(local,global) reference as
    the one-sided version, so it inherits the same reward-hacking fix.
    """
    dO = (O[:, 1:] - O[:, :-1]).pow(2).sum(2)                 # [N,T-1,h,w]
    f = (Mlow[:, 1:] > 0.5) & (Mlow[:, :-1] > 0.5)
    fm = f.float()
    if fm.sum() == 0:
        return O.sum() * 0.0
    dO_det = dO.detach()                                       # ref is a target, not a lever
    pad = ring_k // 2
    dil = F.max_pool2d(fm, ring_k, stride=1, padding=pad)
    ring = ((dil > 0.5) & (~f)).float()
    num = F.avg_pool2d(dO_det * ring, ring_k, stride=1, padding=pad)
    den = F.avg_pool2d(ring, ring_k, stride=1, padding=pad)
    local_ref = num / den.clamp(min=1e-6)

    N, T = O.shape[:2]
    rl = F.interpolate(origs.float().reshape(N * T, 1, *origs.shape[2:]),
                       size=(out_size, out_size), mode="nearest").reshape(N, T, out_size, out_size)
    rmask = (rl != PERSON) & (rl != IGNORE)
    rmask_pair = (rmask[:, 1:] & rmask[:, :-1])
    if rmask_pair.sum() == 0:
        ref = local_ref
    else:
        global_ref = (dO_det * rmask_pair).sum(dim=(2, 3)) / rmask_pair.sum(dim=(2, 3)).clamp(min=1)
        global_ref = global_ref.unsqueeze(-1).unsqueeze(-1).expand_as(local_ref)
        ref = torch.minimum(local_ref, global_ref)             # never more lenient than the global budget

    dev = (dO - ref).pow(2) if two_sided else F.relu(dO - ref)
    return (dev * fm).sum() / fm.sum().clamp(min=1)


def adv_temporal_x(model, loader, device="cuda", epochs=6, lr=1e-3, lam_r=1.0,
                   lam_adv=1.0, lam_tc=0.0, lam_ord=1.0, lr_d=1e-3, out_size=128,
                   replay_loader=None, freeze_bn=False, temporal_kind="conv3d",
                   ring_k=9, dilate_k=9, **_):
    """ADVR-TX: ADVR-T plus a temporal-EXCHANGEABILITY adversary.

    Why go beyond ADVR-T. ADVR-T's temporal term constrains the MAGNITUDE of
    frame-to-frame change in the hole. But magnitude is a proxy: what a
    trajectory attacker actually consumes is whether the change is ORDERED --
    whether frame t+1 follows from frame t the way a moving object would. A
    model can satisfy a magnitude budget and still leave a perfectly readable
    trajectory, simply by moving smoothly and slowly.

    So we add a second adversary, an OrderDiscriminator, which sees the field
    sequence masked to a dilated neighbourhood of the hole and must say whether
    it is in true order or randomly permuted. The model is trained to make that
    judgement impossible. At the optimum the hole's dynamics are temporally
    exchangeable, and a sequence carrying no order information carries no
    trajectory -- which is the property we want, stated directly, rather than
    approached through a magnitude surrogate.

    This also closes the loop with the measurement: the evaluation protocol
    scores leakage as AP(ordered) - AP(time-shuffled), and this objective drives
    exactly that difference to zero. Objective and metric are the same
    functional, so the method is not being graded on a proxy it never optimised.
    """
    from seg.attack import (build_attacker, OrderDiscriminator,
                            clip_person_mask_low, _shuffle_time)

    m = _clone(model).to(device).train()
    if freeze_bn:
        _freeze_bn(m)
    teacher = _frozen(model, device)
    DT = build_attacker(temporal_kind, out_size=out_size).to(device).train()
    DO = OrderDiscriminator().to(device).train()
    optG = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    optD = torch.optim.Adam(list(DT.parameters()) + list(DO.parameters()), lr=lr_d)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)
    replay_it = iter(replay_loader) if replay_loader is not None else None

    def clip_field(zf, N, T):
        p = F.interpolate(F.softmax(zf, 1), size=(out_size, out_size),
                          mode="bilinear", align_corners=False)
        return p.reshape(N, T, *p.shape[1:])

    def hole_view(O, Mlow):
        """Field sequence restricted to a dilated neighbourhood of the hole --
        dilated so the discriminator sees the boundary, which is where a
        silhouette's motion is legible, not just the interior."""
        N, T = Mlow.shape[:2]
        d = F.max_pool2d(Mlow, dilate_k, stride=1, padding=dilate_k // 2)
        return O * d.unsqueeze(2)

    for _e in range(epochs):
        if freeze_bn:
            _freeze_bn(m)
        for imgs, origs, mergeds in loader:
            imgs = imgs.to(device); origs = origs.to(device); mergeds = mergeds.to(device)
            N, T = imgs.shape[:2]
            imgs_f, origs_f, merged_f = _flat(imgs), _flat(origs), _flat(mergeds)
            forget_f, retain_f = _masks(origs_f)
            Mlow = clip_person_mask_low(origs, out_size)

            # ---- adversaries: trajectory reconstruction + order detection ----
            with torch.no_grad():
                O_det = clip_field(seg_forward(m, imgs_f), N, T)
            lossD = bce(DT(O_det), Mlow)
            if T > 1:
                hv = hole_view(O_det, Mlow)
                hv_s, _ = _shuffle_time(hv, Mlow)
                real = DO(hv); fake = DO(hv_s)
                lossD = lossD + bce(real, torch.ones_like(real)) \
                              + bce(fake, torch.zeros_like(fake))
            optD.zero_grad(); lossD.backward(); optD.step()

            # ---- model ----
            zf = seg_forward(m, imgs_f)
            with torch.no_grad():
                z0f = seg_forward(teacher, imgs_f)
            loss = _masked_ce(zf, merged_f, forget_f) + lam_r * _masked_kl(zf, z0f, retain_f)

            O = clip_field(zf, N, T)
            dpred = DT(O); fmask = Mlow > 0.5
            if fmask.any():
                loss = loss + lam_adv * bce(dpred[fmask], torch.zeros_like(dpred[fmask]))
            if lam_tc > 0 and T > 1:
                loss = loss + lam_tc * _temporal_consistency_local(O, Mlow, origs, out_size, ring_k)
            if lam_ord > 0 and T > 1:
                # make the true order look shuffled to DO
                ordered = DO(hole_view(O, Mlow))
                loss = loss + lam_ord * bce(ordered, torch.zeros_like(ordered))

            optG.zero_grad(); loss.backward(); optG.step()

            if replay_it is not None:
                try:
                    rimg, rlab = next(replay_it)
                except StopIteration:
                    replay_it = iter(replay_loader); rimg, rlab = next(replay_it)
                rimg, rlab = rimg.to(device), rlab.to(device)
                optG.zero_grad(); ce(seg_forward(m, rimg), rlab).backward(); optG.step()
    return m


def adv_pool(model, loader, device="cuda", epochs=6, lr=1e-3, lam_r=1.0,
            lam_adv=1.0, lam_tc=0.0, lam_pool=1.0, lr_d=1e-3, out_size=128,
            replay_loader=None, freeze_bn=False, temporal_kind="conv3d",
            ring_k=9, **_):
    """ADVR-P: attacks the mechanism Phase A/B/sec 9 actually found, not the
    one ADVR-T/ADVR-TX were built around. Two changes on top of ADVR-T's
    per-frame blend + retain-KL anchor + retain-replay:

    1. TWO-SIDED L_tc (`_temporal_consistency_local(..., two_sided=True)`,
       reference fixed to be detached + min(local,global) per sec 9). The
       one-sided cap used by ADVR-T/ADVR-TX, relu(dO-ref), is mathematically
       BLIND whenever the hole moves LESS than its reference (verified: a
       frozen hole scores exactly 0.0 under the one-sided version). But the
       mechanistic story behind Delta_redundancy is a FRAME-CONSISTENT bias --
       a hole that barely changes at all is exactly what an averaging attacker
       sharpens into a confident estimate. Two-sided, (dO-ref)^2, requires the
       hole to MATCH context's temporal variance in both directions.

    2. A POOLING discriminator (`PoolDiscriminator`), trained and fooled
       ALONGSIDE the usual reconstruction discriminator DT, replacing
       ADVR-TX's `OrderDiscriminator`. Phase A/B showed order carries no
       measurable signal (synergy = AP(ordered)-AP(shuffled) ~= 0 everywhere,
       including under mamba's full-clip reach) -- OrderDiscriminator was
       attacking a leak that does not exist. PoolDiscriminator is structurally
       incapable of using order at all (it only ever sees O.mean(dim=1)), so
       fooling it directly attacks Delta_redundancy = AP(shuffled) - frame_AP,
       the mechanism that IS real. This is the closest objective-metric match
       attempted so far: what the protocol measures as the leak is
       (approximately) what this discriminator is built to detect.

    DT is kept, not replaced -- it still catches ordinary per-frame/windowed
    reconstruction attempts; D_pool adds a surgical, order-blind check on top.
    """
    from seg.attack import build_attacker, PoolDiscriminator, clip_person_mask_low

    m = _clone(model).to(device).train()
    if freeze_bn:
        _freeze_bn(m)
    teacher = _frozen(model, device)
    DT = build_attacker(temporal_kind, out_size=out_size).to(device).train()
    Dp = PoolDiscriminator().to(device).train()
    optG = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    optD = torch.optim.Adam(list(DT.parameters()) + list(Dp.parameters()), lr=lr_d)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)
    replay_it = iter(replay_loader) if replay_loader is not None else None

    def clip_field(zf, N, T):
        p = F.interpolate(F.softmax(zf, 1), size=(out_size, out_size),
                          mode="bilinear", align_corners=False)
        return p.reshape(N, T, *p.shape[1:])

    for _e in range(epochs):
        if freeze_bn:
            _freeze_bn(m)
        for imgs, origs, mergeds in loader:
            imgs = imgs.to(device); origs = origs.to(device); mergeds = mergeds.to(device)
            N, T = imgs.shape[:2]
            imgs_f, origs_f, merged_f = _flat(imgs), _flat(origs), _flat(mergeds)
            forget_f, retain_f = _masks(origs_f)
            Mlow = clip_person_mask_low(origs, out_size)

            # ---- adversaries: trajectory reconstruction + pooling reconstruction ----
            with torch.no_grad():
                O_det = clip_field(seg_forward(m, imgs_f), N, T)
            lossD = bce(DT(O_det), Mlow)
            pool_pred_det = Dp(O_det)                            # [N,h,w]
            pool_target = (Mlow > 0.5).any(dim=1).float()         # ever-person-in-clip
            lossD = lossD + bce(pool_pred_det, pool_target)
            optD.zero_grad(); lossD.backward(); optD.step()

            # ---- model ----
            zf = seg_forward(m, imgs_f)
            with torch.no_grad():
                z0f = seg_forward(teacher, imgs_f)
            loss = _masked_ce(zf, merged_f, forget_f) + lam_r * _masked_kl(zf, z0f, retain_f)

            O = clip_field(zf, N, T)
            dpred = DT(O); fmask = Mlow > 0.5
            if fmask.any():
                loss = loss + lam_adv * bce(dpred[fmask], torch.zeros_like(dpred[fmask]))

            pool_pred = Dp(O)
            pool_fmask = (Mlow > 0.5).any(dim=1)
            if pool_fmask.any():
                loss = loss + lam_pool * bce(pool_pred[pool_fmask],
                                             torch.zeros_like(pool_pred[pool_fmask]))

            if lam_tc > 0 and T > 1:
                loss = loss + lam_tc * _temporal_consistency_local(
                    O, Mlow, origs, out_size, ring_k, two_sided=True)

            optG.zero_grad(); loss.backward(); optG.step()

            if replay_it is not None:
                try:
                    rimg, rlab = next(replay_it)
                except StopIteration:
                    replay_it = iter(replay_loader); rimg, rlab = next(replay_it)
                rimg, rlab = rimg.to(device), rlab.to(device)
                optG.zero_grad(); ce(seg_forward(m, rimg), rlab).backward(); optG.step()
    return m


CANDIDATES = {
    "ADVR": adv_replay,          # per-frame ADVR (comparison / ablation)
    "ADVR-T": adv_temporal,      # temporal ADVR (global-mean L_tc)
    "ADVR-TX": adv_temporal_x,   # + local-ring L_tc + temporal-exchangeability adversary
    "ADVR-P": adv_pool,          # + pooling discriminator + two-sided L_tc (attacks Delta_redundancy directly)
}


def run_candidate(name, model, loader, device="cuda", **kw):
    if name not in CANDIDATES:
        raise ValueError(f"unknown candidate {name!r}; have {list(CANDIDATES)}")
    return CANDIDATES[name](model, loader, device=device, **kw)
