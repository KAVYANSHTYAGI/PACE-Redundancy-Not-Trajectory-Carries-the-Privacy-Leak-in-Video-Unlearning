"""Selective-state-space (Mamba) temporal mixing for the trajectory attacker.

WHY THE ATTACKER ZOO NEEDED THIS. A leakage result is a claim about the ABSENCE
of recoverable information, so it is only ever as strong as the strongest
attacker that was tried. The existing temporal attackers cannot support a
long-horizon claim:

  * TemporalSilhouetteNet (Conv3d, depth 3, kernel 3) has a measured temporal
    receptive field of +/-3 frames -- span 7. At T=16 it is structurally BLIND
    to 9 of the 16 frames: no amount of training lets it integrate a 16-frame
    trajectory, because the information never reaches the output unit. Reporting
    "no leakage at T=16" from this attacker is close to vacuous.
  * ConvGRUAttacker is bidirectional and therefore unbounded in principle, but a
    gated RNN's memory decays multiplicatively, so long-range evidence is
    attenuated in practice.

A selective SSM is the right instrument for the specific thing being claimed:
linear-time in sequence length, with an input-dependent (selective) state
transition that can hold or discard evidence over arbitrarily long spans. If a
Mamba attacker also finds nothing at T=16, "no long-range trajectory leakage"
becomes a claim about the model rather than about the probe's receptive field.

Two backends, chosen automatically:
  * the official `mamba_ssm.Mamba` (fused CUDA selective scan) when available
    and running on CUDA -- this is what the experiments use;
  * `S6Ref`, a dependency-free PyTorch selective scan, otherwise. Clip lengths
    here are <= 16, so the sequential scan is cheap, and it keeps the apparatus
    runnable/testable on CPU and reproducible without a compiled extension.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:                                    # fused CUDA kernel, optional
    from mamba_ssm import Mamba as _MambaCUDA
    _HAS_MAMBA_SSM = True
except Exception:                       # pragma: no cover
    _MambaCUDA = None
    _HAS_MAMBA_SSM = False


class S6Ref(nn.Module):
    """Reference selective-scan block (Mamba's S6) in pure PyTorch.

    h_t = exp(dt_t * A) h_{t-1} + dt_t * B_t * x_t,   y_t = C_t . h_t + D * x_t

    with (dt_t, B_t, C_t) predicted from x_t -- the "selective" part, which is
    what lets the state keep evidence about one frame while ignoring others.
    Scanned sequentially; fine at T <= 16.
    """

    def __init__(self, d_model, d_state=8, d_conv=4, expand=2,
                 dt_min=1e-3, dt_max=0.1):
        super().__init__()
        d_in = expand * d_model
        self.d_model, self.d_state, self.d_in = d_model, d_state, d_in
        self.dt_rank = max(1, d_model // 16)
        self.in_proj = nn.Linear(d_model, 2 * d_in, bias=False)
        self.conv1d = nn.Conv1d(d_in, d_in, d_conv, padding=d_conv - 1, groups=d_in)
        self.d_conv = d_conv
        self.x_proj = nn.Linear(d_in, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_in, bias=True)
        dt = torch.exp(torch.rand(d_in) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=1e-4)
        with torch.no_grad():                       # inverse-softplus init
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_in, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_in))
        self.out_proj = nn.Linear(d_in, d_model, bias=False)

    def forward(self, x):                            # x: [B, L, d_model]
        B_, L, _ = x.shape
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)                   # [B,L,d_in] each
        u = self.conv1d(u.transpose(1, 2))[..., :L].transpose(1, 2)
        u = F.silu(u)

        A = -torch.exp(self.A_log)                   # [d_in, n]
        proj = self.x_proj(u)
        dtr, Bm, Cm = torch.split(
            proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dtr))        # [B,L,d_in]

        h = u.new_zeros(B_, self.d_in, self.d_state)
        ys = []
        for t in range(L):                           # sequential scan, L<=16
            dt_t = delta[:, t].unsqueeze(-1)         # [B,d_in,1]
            dA = torch.exp(dt_t * A)                 # [B,d_in,n]
            dBx = dt_t * Bm[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
            h = dA * h + dBx
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, 1) + u * self.D          # [B,L,d_in]
        return self.out_proj(y * F.silu(z))


class BiMamba(nn.Module):
    """Bidirectional selective-SSM mixing along time, pre-norm + residual.

    Bidirectional because the attacker is an offline adversary reading a whole
    stored clip -- it has no causality constraint, and the per-frame mask at t is
    equally constrained by frames before and after t. Matches ConvGRUAttacker's
    bidirectional treatment so the comparison isolates the state-space mechanism
    rather than the direction of information flow.
    """

    def __init__(self, d_model, d_state=8, d_conv=4, expand=2, backend="auto"):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        use_cuda_kernel = (backend == "cuda"
                           or (backend == "auto" and _HAS_MAMBA_SSM))
        if use_cuda_kernel and not _HAS_MAMBA_SSM:
            raise RuntimeError("backend='cuda' but mamba_ssm is not importable")
        mk = (lambda: _MambaCUDA(d_model=d_model, d_state=d_state,
                                 d_conv=d_conv, expand=expand)) if use_cuda_kernel \
            else (lambda: S6Ref(d_model, d_state=d_state, d_conv=d_conv, expand=expand))
        self.fwd = mk()
        self.bwd = mk()
        self.backend = "mamba_ssm" if use_cuda_kernel else "s6ref"

    def forward(self, x):                            # [B, L, d]
        h = self.norm(x)
        return x + self.fwd(h) + self.bwd(h.flip(1)).flip(1)
