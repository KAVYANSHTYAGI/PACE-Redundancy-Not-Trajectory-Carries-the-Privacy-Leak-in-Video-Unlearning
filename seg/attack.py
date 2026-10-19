"""Silhouette-reconstruction attackers for VSPW person-unlearning.

Two families, and the contrast between them IS the paper:

  PER-FRAME attackers (SilhouetteNet / UNet / ViT) -- ported verbatim from
  project1. Given a single frame's output field O_t, reconstruct the person mask
  M_t. This measures spatial silhouette leakage I(M_t; O_t), exactly as in the
  image paper.

  TEMPORAL attackers (TemporalSilhouetteNet / ConvGRUAttacker) -- new. Given a
  WINDOW of output fields O_{t-w..t+w}, reconstruct the mask TRAJECTORY M_{t-w..t+w}.
  These can exploit cross-frame continuity (a person moves smoothly; a random
  blend does not) that no per-frame attacker sees. This measures temporal
  silhouette leakage I(M_{1:T}; O_{1:T}) -- POSITIONING.md's central quantity.

The oracle (retrained never seeing person) gives the irreducible context-prior
floor on BOTH; a method "leaks temporally" iff its temporal attacker beats the
temporal floor even when its per-frame attacker sits at the per-frame floor.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from seg.data import PERSON, NUM_CLASSES
from seg.model import seg_forward

FEAT_DIM = 256


# ===================== PER-FRAME attackers (ported) ======================== #
class SilhouetteNet(nn.Module):
    """Dilated-conv per-frame silhouette attacker. width/depth/dilation control
    capacity for the capacity-sweep / transfer rebuttal experiments."""
    def __init__(self, c_in=NUM_CLASSES, width=64, depth=3, dilation=1):
        super().__init__()
        layers, cin = [], c_in
        for _ in range(depth):
            layers += [nn.Conv2d(cin, width, 3, padding=dilation, dilation=dilation),
                       nn.BatchNorm2d(width), nn.ReLU(True)]
            cin = width
        layers += [nn.Conv2d(width, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, p):
        return self.net(p)[:, 0]                       # [N,h,w] logits


class _DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class UNetAttacker(nn.Module):
    """U-Net per-frame reconstructor with skip connections (~17M at base=96)."""
    def __init__(self, c_in=NUM_CLASSES, base=64):
        super().__init__()
        b = base
        self.inc = _DoubleConv(c_in, b)
        self.d1 = _DoubleConv(b, b * 2); self.d2 = _DoubleConv(b * 2, b * 4)
        self.d3 = _DoubleConv(b * 4, b * 8); self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2); self.u3 = _DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2); self.u2 = _DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2); self.u1 = _DoubleConv(b * 2, b)
        self.outc = nn.Conv2d(b, 1, 1)

    def forward(self, p):
        x0 = self.inc(p); x1 = self.d1(self.pool(x0))
        x2 = self.d2(self.pool(x1)); x3 = self.d3(self.pool(x2))
        y = self.u3(torch.cat([self.up3(x3), x2], 1))
        y = self.u2(torch.cat([self.up2(y), x1], 1))
        y = self.u1(torch.cat([self.up1(y), x0], 1))
        return self.outc(y)[:, 0]


class ViTAttacker(nn.Module):
    """Patch-transformer per-frame reconstructor (global attention from layer 1)."""
    def __init__(self, c_in=NUM_CLASSES, img=128, patch=8, dim=384, depth=8, heads=6, mlp=4):
        super().__init__()
        self.img, self.patch = img, patch
        self.gh = img // patch
        self.proj = nn.Conv2d(c_in, dim, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.gh * self.gh, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * mlp, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch * patch)

    def forward(self, p):
        if p.shape[-1] != self.img:
            p = F.interpolate(p, size=(self.img, self.img), mode="bilinear", align_corners=False)
        t = self.proj(p).flatten(2).transpose(1, 2)
        t = self.enc(t + self.pos)
        t = self.head(self.norm(t))
        gh, ps = self.gh, self.patch
        t = t.reshape(-1, gh, gh, ps, ps).permute(0, 1, 3, 2, 4)
        return t.reshape(-1, gh * ps, gh * ps)


# ===================== TEMPORAL attackers (new) ============================ #
class TemporalSilhouetteNet(nn.Module):
    """3D-conv trajectory reconstructor. Input a window of output fields
    [N,T,C,h,w]; a stack of spatiotemporal Conv3d layers gives each frame a
    receptive field spanning +/- (depth) neighbouring frames, so the head can use
    trajectory continuity a per-frame conv cannot. Output [N,T,h,w] mask logits.

    This is the minimal, dependency-free temporal attacker (no optical-flow model):
    the Conv3d kernel itself learns whatever motion cue is present in the stacked
    fields. width/depth control capacity for the temporal capacity sweep."""
    def __init__(self, c_in=NUM_CLASSES, width=64, depth=3, dilation=1):
        super().__init__()
        layers, cin = [], c_in
        for _ in range(depth):
            layers += [nn.Conv3d(cin, width, 3, padding=dilation, dilation=dilation),
                       nn.BatchNorm3d(width), nn.ReLU(True)]
            cin = width
        layers += [nn.Conv3d(width, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, p):                              # p: [N,T,C,h,w]
        p = p.permute(0, 2, 1, 3, 4)                   # -> [N,C,T,h,w] for Conv3d
        y = self.net(p)[:, 0]                          # [N,T,h,w]
        return y


class VideoMambaAttacker(nn.Module):
    """Selective-SSM trajectory attacker: unbounded temporal reach, linear cost.

    Alternates per-frame SPATIAL mixing with per-pixel TEMPORAL mixing. Both are
    needed and neither suffices: a purely per-pixel scan along time cannot track
    a person who moves across pixels, and a purely spatial network cannot see
    across frames at all. The spatial stage carries evidence to where the object
    went; the SSM stage carries it across time.

    The scan runs at stride-`stride` resolution (128 -> 32 by default) because
    trajectory evidence is coarse, and because a full-resolution scan would hold
    H*W independent sequences per clip. The head upsamples back to out_size so
    the output is directly comparable with the Conv3d and ConvGRU attackers.

    Compare against TemporalSilhouetteNet, whose measured receptive field is
    +/-3 frames: at T=16 that attacker cannot integrate more than 7 of the 16
    frames, so its null result is a statement about its own reach. This one has
    no such ceiling, which is what makes a long-horizon "no leakage" claim
    meaningful.
    """

    def __init__(self, c_in=NUM_CLASSES, d=64, depth=2, d_state=8, expand=2,
                 out_size=128, stride=4, backend="auto"):
        super().__init__()
        from seg.mamba import BiMamba
        self.out_size = out_size
        s1 = 2 if stride >= 2 else 1
        s2 = 2 if stride >= 4 else 1
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, d, 3, stride=s1, padding=1), nn.BatchNorm2d(d), nn.ReLU(True),
            nn.Conv2d(d, d, 3, stride=s2, padding=1), nn.BatchNorm2d(d), nn.ReLU(True))
        self.spatial = nn.ModuleList([
            nn.Sequential(nn.Conv2d(d, d, 3, padding=1, groups=d),
                          nn.Conv2d(d, d, 1), nn.BatchNorm2d(d), nn.ReLU(True))
            for _ in range(depth)])
        self.temporal = nn.ModuleList([
            BiMamba(d, d_state=d_state, expand=expand, backend=backend)
            for _ in range(depth)])
        self.head = nn.Conv2d(d, 1, 1)
        self.backend = self.temporal[0].backend

    def forward(self, p):                              # p: [N,T,C,h,w]
        N, T = p.shape[:2]
        x = self.stem(p.reshape(N * T, *p.shape[2:]))  # [N*T,d,hs,ws]
        d, hs, ws = x.shape[1:]
        for sp, tp in zip(self.spatial, self.temporal):
            x = x + sp(x)                              # per-frame spatial mixing
            # [N*T,d,hs,ws] -> one length-T sequence per spatial position
            s = x.reshape(N, T, d, hs, ws).permute(0, 3, 4, 1, 2).reshape(N * hs * ws, T, d)
            s = tp(s)
            x = s.reshape(N, hs, ws, T, d).permute(0, 3, 4, 1, 2).reshape(N * T, d, hs, ws)
        y = self.head(x)                               # [N*T,1,hs,ws]
        y = F.interpolate(y, size=(self.out_size, self.out_size),
                          mode="bilinear", align_corners=False)
        return y.reshape(N, T, self.out_size, self.out_size)


class OrderDiscriminator(nn.Module):
    """P(this field sequence is in its TRUE temporal order), read off the forget
    region only.

    Motivation. A trajectory is recoverable from the forgotten region exactly
    when that region's field sequence still carries information about time
    ORDER. If the sequence restricted to the hole were temporally exchangeable
    -- statistically indistinguishable from a random permutation of itself --
    then by definition no attacker, of any capacity, could read a trajectory off
    it; the best it could do is per-frame work plus order-free pooling. So
    "order-indistinguishable inside the hole" is the exact operationalisation of
    "no temporal silhouette leakage", and this module is the estimator for it.

    Used two ways, deliberately symmetric: as the training adversary in
    seg.candidates.adv_temporal_x, and (via the time-shuffled arm of
    scripts/run_temporal_protocol.py) as the yardstick at evaluation time -- the
    objective optimises the same functional the metric reports.
    """

    def __init__(self, c_in=NUM_CLASSES, width=48, depth=3):
        super().__init__()
        layers, cin = [], c_in
        for i in range(depth):
            layers += [nn.Conv3d(cin, width, 3, padding=1),
                       nn.BatchNorm3d(width), nn.ReLU(True)]
            if i < depth - 1:
                layers += [nn.AvgPool3d((1, 2, 2))]
            cin = width
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(width, 1)

    def forward(self, p):                              # p: [N,T,C,h,w]
        y = self.net(p.permute(0, 2, 1, 3, 4))         # [N,W,T,h',w']
        y = y.mean(dim=(2, 3, 4))                      # pool time+space -> [N,W]
        return self.head(y)[:, 0]                      # [N] logits


class PoolDiscriminator(nn.Module):
    """Order-invariant leakage judge: sees ONLY the temporal MEAN of the field
    sequence, O.mean(dim=1), collapsed before any learned parameter touches it.

    Motivation (2026-07-30, see EXPERIMENTS.md sec 5/9). Phase A/B showed the
    real leak is not order/trajectory (synergy = AP(ordered) - AP(shuffled) is
    ~0 everywhere, including under mamba's full-clip reach) -- it is REDUNDANCY:
    an attacker that does nothing but average per-frame estimates recovers far
    more than a single-frame attacker (Delta_redundancy = AP(shuffled) -
    frame_AP, large and real for ADVR/baselines, much smaller for ADVR-T). Every
    adversary trained against so far (TemporalSilhouetteNet, ConvGRUAttacker,
    VideoMambaAttacker, OrderDiscriminator) is an order-SENSITIVE architecture,
    so none of them was ever forced to specifically attack the pooling
    mechanism -- ADVR-T's defense against it was the indirect side-effect of a
    magnitude cap (L_tc), not a direct adversarial target.

    This module IS the direct target: it is structurally incapable of using
    frame order (mean is a symmetric function of its inputs), so fooling it can
    only be accomplished by removing whatever FRAME-CONSISTENT bias would
    otherwise survive averaging -- exactly Delta_redundancy's mechanism, stated
    directly rather than approached through TemporalSilhouetteNet's limited
    receptive field or OrderDiscriminator's now-known-irrelevant target.

    Produces ONE prediction per clip (not one per frame); callers compare it
    against every frame's own mask (broadcast), mirroring exactly how
    eval_temporal_attacker scores the SHUFFLED arm: one piece of aggregate
    evidence, judged against each frame's ground truth.
    """

    def __init__(self, c_in=NUM_CLASSES, width=64, depth=3, dilation=1):
        super().__init__()
        self.net = SilhouetteNet(c_in=c_in, width=width, depth=depth, dilation=dilation)

    def forward(self, O):                               # O: [N,T,C,h,w]
        pooled = O.mean(dim=1)                           # [N,C,h,w], order-invariant by construction
        return self.net(pooled)                          # [N,h,w] logits, one map per clip


class DeepSetsAttacker(nn.Module):
    """Learned permutation-invariant attacker: D(O_1:T) = rho(mean_t phi(O_t)).

    Strictly more expressive than PoolDiscriminator, which pools the raw
    field (phi = identity) before any learned parameter touches it. Here phi
    is a learned per-frame CNN embedding, applied identically and
    independently to every frame (shared weights, no cross-frame mixing),
    so the representation summed over time is richer than a raw average of
    logits while remaining EXACTLY permutation-invariant by the same
    argument as the pooled adversary in the main paper (Eq. 8): the mean is
    commutative regardless of what phi computes per frame, since phi is
    applied before pooling and never sees more than one frame at a time.

    This tests whether PACE's suppression is specific to being fooling a
    fixed-mean-of-raw-logits adversary, or whether it survives a strictly
    more expressive order-invariant attacker.
    """

    def __init__(self, c_in=NUM_CLASSES, width=64, depth=3, embed_dim=64):
        super().__init__()
        layers = []
        c = c_in
        for _ in range(depth):
            layers += [nn.Conv2d(c, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(True)]
            c = width
        layers += [nn.Conv2d(width, embed_dim, 1)]
        self.phi = nn.Sequential(*layers)                 # per-frame embedding
        self.rho = SilhouetteNet(c_in=embed_dim, width=width, depth=2)  # set decoder

    def forward(self, O):                                 # O: [N,T,C,h,w]
        N, T = O.shape[:2]
        flat = O.reshape(N * T, *O.shape[2:])
        z = self.phi(flat)                                # [N*T,embed_dim,h,w]
        z = z.reshape(N, T, *z.shape[1:]).mean(dim=1)      # order-invariant pool
        out = self.rho(z)                                 # [N,h,w] logits, one map per clip
        return out.unsqueeze(1).expand(-1, T, -1, -1)      # broadcast for per-frame scoring


class _ConvGRUCell(nn.Module):
    def __init__(self, c_in, c_hid, k=3):
        super().__init__()
        p = k // 2
        self.convz = nn.Conv2d(c_in + c_hid, c_hid, k, padding=p)
        self.convr = nn.Conv2d(c_in + c_hid, c_hid, k, padding=p)
        self.convq = nn.Conv2d(c_in + c_hid, c_hid, k, padding=p)

    def forward(self, x, h):
        xh = torch.cat([x, h], 1)
        z = torch.sigmoid(self.convz(xh)); r = torch.sigmoid(self.convr(xh))
        q = torch.tanh(self.convq(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * q


class ConvGRUAttacker(nn.Module):
    """Bidirectional ConvGRU trajectory reconstructor -- an explicitly recurrent
    temporal attacker (contrast with the Conv3d one, which is a fixed-window
    convolution). Processes the field sequence forward and backward, so every
    frame's prediction can use the whole clip. A genuinely different temporal
    inductive bias for the attacker-zoo rebuttal."""
    def __init__(self, c_in=NUM_CLASSES, hid=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(c_in, hid, 3, padding=1), nn.ReLU(True))
        self.fwd = _ConvGRUCell(hid, hid)
        self.bwd = _ConvGRUCell(hid, hid)
        self.head = nn.Sequential(nn.Conv2d(2 * hid, hid, 3, padding=1),
                                  nn.ReLU(True), nn.Conv2d(hid, 1, 1))

    def forward(self, p):                              # p: [N,T,C,h,w]
        N, T, C, h, w = p.shape
        feats = [self.enc(p[:, t]) for t in range(T)]
        hid = feats[0].shape[1]
        hf = torch.zeros(N, hid, h, w, device=p.device)
        fout = []
        for t in range(T):
            hf = self.fwd(feats[t], hf); fout.append(hf)
        hb = torch.zeros(N, hid, h, w, device=p.device)
        bout = [None] * T
        for t in range(T - 1, -1, -1):
            hb = self.bwd(feats[t], hb); bout[t] = hb
        outs = [self.head(torch.cat([fout[t], bout[t]], 1))[:, 0] for t in range(T)]
        return torch.stack(outs, 1)                    # [N,T,h,w]


# ============================ factory / utils ============================== #
def build_attacker(kind="cnn", c_in=NUM_CLASSES, width=64, depth=3, dilation=1,
                   base=64, dim=384, vdepth=8, heads=6, out_size=128, hid=64,
                   d_state=8, mamba_depth=2, mamba_stride=4, backend="auto"):
    """Factory. Per-frame: cnn/unet/vit. Temporal: conv3d/convgru/mamba.

    Temporal reach differs qualitatively across the temporal kinds, which is the
    point of keeping all three: conv3d is capped at +/-3 frames (measured),
    convgru is unbounded but decays, mamba is unbounded and selective.
    """
    if kind == "cnn":
        return SilhouetteNet(c_in=c_in, width=width, depth=depth, dilation=dilation)
    if kind == "unet":
        return UNetAttacker(c_in=c_in, base=base)
    if kind == "vit":
        return ViTAttacker(c_in=c_in, img=out_size, dim=dim, depth=vdepth, heads=heads)
    if kind == "conv3d":
        return TemporalSilhouetteNet(c_in=c_in, width=width, depth=depth, dilation=dilation)
    if kind == "convgru":
        return ConvGRUAttacker(c_in=c_in, hid=hid)
    if kind == "mamba":
        return VideoMambaAttacker(c_in=c_in, d=hid, depth=mamba_depth,
                                  d_state=d_state, out_size=out_size,
                                  stride=mamba_stride, backend=backend)
    if kind == "deepsets":
        return DeepSetsAttacker(c_in=c_in, width=width, depth=depth)
    raise ValueError(f"unknown attacker kind {kind!r}")


TEMPORAL_KINDS = {"conv3d", "convgru", "mamba", "deepsets"}


def n_params(module):
    return sum(p.numel() for p in module.parameters())


# ============================ field extraction ============================= #
class _AttackFeatHook:
    def __init__(self, model):
        self.feat = None
        self.h = model.classifier[4].register_forward_pre_hook(
            lambda m, inp: setattr(self, "feat", inp[0]))

    def close(self):
        self.h.remove()


@torch.no_grad()
def output_field(model, img, transform=None, out_size=128):
    z = seg_forward(model, img)                        # [N,C,H,W]
    if transform == "mask":                            # DELETE-style: drop person logit
        z = z.clone(); z[:, PERSON] = -1e4
    p = F.softmax(z, 1)
    return F.interpolate(p, size=(out_size, out_size), mode="bilinear", align_corners=False)


@torch.no_grad()
def feature_field(model, img, out_size=128):
    hook = _AttackFeatHook(model)
    seg_forward(model, img)
    feat = hook.feat
    hook.close()
    return F.interpolate(feat, size=(out_size, out_size), mode="bilinear", align_corners=False)


def person_mask_low(orig, out_size=128):
    M = (orig == PERSON).float().unsqueeze(1)
    return F.interpolate(M, size=(out_size, out_size), mode="nearest")[:, 0]


def _field(model, img, transform, out_size, feat):
    return feature_field(model, img, out_size) if feat \
        else output_field(model, img, transform, out_size)


@torch.no_grad()
def clip_output_field(model, imgs, transform=None, out_size=128, feat=False):
    """imgs [N,T,3,H,W] -> field stack [N,T,C,h,w]. Runs the (image) model on each
    frame independently -- the temporal signal we probe lives in the SEQUENCE of
    per-frame fields, not in any temporal model component (the B0 model is
    per-frame). This is exactly the point: leakage can be temporal even when the
    model is not."""
    N, T = imgs.shape[:2]
    flat = imgs.reshape(N * T, *imgs.shape[2:])
    p = _field(model, flat, transform, out_size, feat)  # [N*T,C,h,w]
    return p.reshape(N, T, *p.shape[1:])


def clip_person_mask_low(origs, out_size=128):
    """origs [N,T,H,W] -> [N,T,h,w] person masks."""
    N, T = origs.shape[:2]
    M = person_mask_low(origs.reshape(N * T, *origs.shape[2:]), out_size)
    return M.reshape(N, T, *M.shape[1:])


# ===================== per-frame train / eval (ported) ==================== #
def train_attacker(model, loader, device, transform=None, epochs=2, out_size=128,
                   max_iters=400, width=64, depth=3, dilation=1, feat=False,
                   kind="cnn", lr=1e-3, base=64, dim=384, vdepth=8, heads=6, D=None):
    c_in = FEAT_DIM if feat else NUM_CLASSES
    if D is None:
        D = build_attacker(kind, c_in=c_in, width=width, depth=depth, dilation=dilation,
                           base=base, dim=dim, vdepth=vdepth, heads=heads,
                           out_size=out_size).to(device)
    D.train(); model.eval()
    opt = torch.optim.Adam(D.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    it = 0
    for _e in range(epochs):
        for img, orig, _m in loader:
            p = _field(model, img.to(device), transform, out_size, feat)
            M = person_mask_low(orig.to(device), out_size)
            loss = bce(D(p), M)
            opt.zero_grad(); loss.backward(); opt.step()
            it += 1
            if it >= max_iters:
                return D
    return D


@torch.no_grad()
def eval_attacker(D, model, loader, device, transform=None, out_size=128, feat=False):
    """Per-frame reconstruction score. Same threshold-free scoring as the
    temporal case so the two surfaces are directly comparable -- comparing an
    IoU@0.5 on one surface against an IoU@0.5 on the other is only meaningful if
    both attackers happen to be equally well calibrated, which they are not."""
    from seg.metrics import ScoreHist
    D.eval(); model.eval()
    hist = ScoreHist(device=device)
    for img, orig, _m in loader:
        p = _field(model, img.to(device), transform, out_size, feat)
        M = person_mask_low(orig.to(device), out_size)
        hist.add(torch.sigmoid(D(p)), M)
    return hist.metrics(prefix="attacker_")


# ===================== temporal train / eval (new) ======================== #
def train_temporal_attacker(model, loader, device, transform=None, epochs=2,
                            out_size=128, max_iters=400, width=64, depth=3,
                            dilation=1, feat=False, kind="conv3d", lr=1e-3,
                            hid=64, D=None, accum_steps=1, shuffle_time=False):
    """Train a temporal attacker on VSPWClip batches (imgs[N,T,3,H,W], origs, _).
    The attacker reconstructs the person-mask trajectory M[N,T,h,w] from the
    field-sequence O[N,T,C,h,w].

    `accum_steps` accumulates gradients over that many loader batches per
    optimizer step. This exists to keep the ATTACKER'S BUDGET MATCHED across
    clip lengths: memory forces the physical batch down as T grows (bs 8 at T=2
    but bs 1 at T=16), so a fixed max_iters silently handed the attacker 8x less
    data and far noisier gradients at large T. Leakage then falls with T for
    every model -- including the oracle floor, which cannot actually depend on T
    -- and the fall is an artifact of the measurement, not of the defense. Drive
    this from the caller so every T does the same number of optimizer steps at
    the same effective batch size in clips.

    `shuffle_time` trains the attacker on time-permuted clips (see _shuffle_time)
    -- the control that separates trajectory-borne leakage from mere denoising.

    Records a convergence trace on the returned module (`D.train_losses`): a
    leakage claim is a claim about absent information, so we must be able to
    tell "the attacker converged and found nothing" from "the attacker never
    trained".
    """
    c_in = FEAT_DIM if feat else NUM_CLASSES
    if D is None:
        D = build_attacker(kind, c_in=c_in, width=width, depth=depth,
                           dilation=dilation, out_size=out_size, hid=hid).to(device)
    D.train(); model.eval()
    opt = torch.optim.Adam(D.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    losses = []
    steps = 0
    micro = 0
    opt.zero_grad()
    for _e in range(epochs):
        for imgs, origs, _m in loader:
            O = clip_output_field(model, imgs.to(device), transform, out_size, feat)
            M = clip_person_mask_low(origs.to(device), out_size)
            if shuffle_time:
                O, M = _shuffle_time(O, M)
            loss = bce(D(O), M) / accum_steps
            loss.backward()
            losses.append(float(loss.item()) * accum_steps)
            micro += 1
            if micro % accum_steps == 0:
                opt.step(); opt.zero_grad(); steps += 1
                if steps >= max_iters:
                    D.train_losses = losses
                    return D
    D.train_losses = losses
    return D


@torch.no_grad()
def eval_temporal_attacker(D, model, loader, device, transform=None, out_size=128,
                           feat=False, shuffle_time=False):
    """Trajectory reconstruction score, threshold-free.

    Returns AP (primary), AUC, IoU@0.5, IoU_best, pos_rate and traj_consistency.
    Scored exactly over every pixel via seg.metrics.ScoreHist rather than
    IoU@0.5 on a 20k subsample -- see that module for why the old scoring could
    report 0.0 leakage for a demonstrably informative attacker.

    `shuffle_time` evaluates on time-permuted clips; pair it with a
    shuffle-trained attacker to measure how much of the leakage survives when
    trajectory order is destroyed.
    """
    from seg.metrics import ScoreHist, TrajConsistency
    D.eval(); model.eval()
    hist = ScoreHist(device=device)
    tc = TrajConsistency()
    for imgs, origs, _m in loader:
        O = clip_output_field(model, imgs.to(device), transform, out_size, feat)
        M = clip_person_mask_low(origs.to(device), out_size)
        if shuffle_time:
            O, M = _shuffle_time(O, M)
        prob = torch.sigmoid(D(O))                     # [N,T,h,w]
        hist.add(prob, M)
        tc.add(prob > 0.5)
    out = hist.metrics(prefix="attacker_")
    out["traj_consistency"] = tc.value()
    return out


def _auc_iou(inter, union, ys, ps):
    """DEPRECATED subsampled scorer, kept so older call sites keep working.
    New code should use seg.metrics.ScoreHist (exact, threshold-free) -- see
    that module's docstring for why IoU@0.5 alone is unsafe here."""
    iou = 100 * inter / max(union, 1)
    y = np.concatenate(ys); pscore = np.concatenate(ps)
    valid = np.isfinite(pscore)
    if valid.sum() < 10 or not (y[valid] > 0.5).any() or not (y[valid] <= 0.5).any():
        auc = float("nan")
    else:
        auc = roc_auc_score((y[valid] > 0.5).astype(int), pscore[valid])
    return {"attacker_IoU": iou, "attacker_AUC": 100 * auc}


def _shuffle_time(O, M, generator=None):
    """Apply the SAME random time-permutation to a clip's field sequence and its
    mask trajectory, independently per clip in the batch.

    This is the control that isolates the paper's actual claim. Two very
    different mechanisms both raise temporal-attacker scores above a per-frame
    attacker's:

      (1) DENOISING / redundancy. The mask is temporally correlated, so a
          temporal attacker can average T noisy per-frame estimates and cut its
          variance. This is order-INDEPENDENT and is not interesting -- any
          per-frame-leaky model exhibits it, and it grows with T for free.
      (2) SYNERGY. Information carried by the trajectory itself (motion
          signature, differential cues) that no single frame contains. This is
          order-DEPENDENT and is the claim "temporal silhouette leakage" makes.

    Shuffling time destroys (2) while leaving (1) completely intact -- the same
    T per-frame fields, in a different order, still average identically. So:
    score(ordered) - score(shuffled) is the part of temporal leakage that is
    genuinely trajectory-borne. Without this control, a temporal-vs-frame gap is
    fully explained by (1) and the paper's central claim is unsupported.
    """
    N, T = O.shape[:2]
    if T < 2:
        return O, M
    idx = torch.stack([torch.randperm(T, generator=generator, device="cpu")
                       for _ in range(N)]).to(O.device)
    oi = idx.view(N, T, *([1] * (O.dim() - 2))).expand_as(O)
    mi = idx.view(N, T, *([1] * (M.dim() - 2))).expand_as(M)
    return O.gather(1, oi), M.gather(1, mi)
