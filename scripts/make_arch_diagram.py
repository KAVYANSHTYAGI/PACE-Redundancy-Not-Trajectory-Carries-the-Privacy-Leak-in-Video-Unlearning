"""Full-width method figure: the three-step progression of the method line,
ADVR (per-frame adversary) -> ADVR-T (temporal adversary + one-sided L_tc)
-> ADVR-P (order-invariant pooled adversary + two-sided L_tc), with the shared
generator/objective spelled out once. Vector PDF sized for a two-column span.

Usage: python scripts/make_arch_diagram.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"]})

C_FRAME = "#e9edf2"
C_GEN = "#cfe3f5"
C_ADV = "#f6dcdc"
C_POOL = "#fbeecb"
C_NEW = "#d5ecdd"
C_TXT = "#141414"
EDGE = "#2f2f2f"


def box(ax, x, y, w, h, text, fc, fs=7.6, weight="normal", ec=EDGE, lw=1.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=fc, edgecolor=ec,
                                linewidth=lw, linestyle=ls,
                                boxstyle="round,pad=0.10,rounding_size=0.06"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight, color=C_TXT, linespacing=1.35)


def arr(ax, p0, p1, color=EDGE, lw=1.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9,
                                 color=color, lw=lw, linestyle=ls,
                                 shrinkA=1, shrinkB=1))


def panel(ax, title, subtitle, adv_label, adv_color, ltc_text, ltc_color,
          highlight=False, strike_order=False):
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.2); ax.axis("off")

    tc = "#0b4f6c" if highlight else C_TXT
    ax.text(5.0, 6.75, title, ha="center", fontsize=9.4, weight="bold", color=tc)
    ax.text(5.0, 6.18, subtitle, ha="center", fontsize=6.6, color="#4a4a4a",
            style="italic")

    # shared trunk: clip -> generator -> outputs
    box(ax, 0.15, 4.70, 2.55, 0.95, "clip $I_{1:T}$", C_FRAME, fs=6.6)
    box(ax, 3.25, 4.70, 3.50, 0.95, "generator $G$", C_GEN, fs=6.6, weight="bold")
    box(ax, 7.35, 4.70, 2.40, 0.95, "$O_{1:T}$", C_GEN, fs=6.6)
    arr(ax, (2.72, 5.17), (3.22, 5.17))
    arr(ax, (6.78, 5.17), (7.33, 5.17))

    # adversary branch
    box(ax, 0.35, 2.45, 9.30, 1.55, adv_label, adv_color, fs=6.5,
        weight="bold" if highlight else "normal",
        lw=1.7 if highlight else 1.0)
    arr(ax, (8.30, 4.68), (6.60, 4.03))
    ax.text(8.45, 4.28, "adv.", fontsize=6.0, color="#555555")

    # consistency term
    box(ax, 0.35, 0.70, 9.30, 1.30, ltc_text, ltc_color, fs=6.5,
        lw=1.7 if highlight else 1.0)
    arr(ax, (5.00, 2.43), (5.00, 2.02), ls="--", color="#5b3aa0")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55))

    panel(axes[0], "PACE-F", "frame adversary",
          "per-frame discriminator $D_F$\nsees each $O_t$ independently\n"
          "$\\Rightarrow$ each frame hard to attack",
          C_ADV, "no temporal term", "#eeeeee")

    panel(axes[1], "PACE-T", "temporal adversary, one-sided cap",
          "temporal discriminator $D_T$\nreads the $\\it{ordered}$ sequence\n"
          "PACE-TO adds an order adversary\n(removed in PACE: order carries no signal)",
          C_ADV,
          "one-sided $L_{tc}=\\mathrm{relu}(\\Delta O-\\mathrm{ref})$\n"
          "penalises excess motion only;\na frozen region scores exactly 0",
          "#ede4fb")

    panel(axes[2], "PACE (ours)", "order-invariant pooled adversary",
          "PoolDiscriminator $D_P$ sees ONLY\n$\\bar{O}=\\mathrm{mean}_t(O_t)$\n"
          "exactly permutation-invariant;\nattacks pooled redundancy directly",
          C_NEW,
          "two-sided $L_{tc}=(\\Delta O-\\mathrm{ref})^{2}$\n"
          "penalises deficient motion too\n(closes the frozen-region loophole)",
          "#d9f2e6", highlight=True)

    fig.text(0.5, 0.015,
             r"shared objective:  $L_G=\lambda_r L_r+\lambda_{adv}\,"
             r"(\mathrm{fool\ adversaries})+\lambda_{tc}L_{tc}$",
             ha="center", fontsize=7.0, color="#333333")

    plt.tight_layout(rect=[0, 0.045, 1, 1])
    os.makedirs("paper/figures", exist_ok=True)
    p = "paper/figures/advrp_architecture.pdf"
    plt.savefig(p, bbox_inches="tight"); plt.close()
    print("wrote", p)


if __name__ == "__main__":
    main()
