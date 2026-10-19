"""Paper figures beyond the qualitative grids and the architecture diagram:

  fig_teaser.pdf    conceptual: why per-frame audits miss pooled leakage, and
                    why time-shuffling separates redundancy from trajectory
  fig_frontier.pdf  privacy-utility scatter over all 12 car-class models
  fig_redundancy.pdf  Delta_redundancy per model across three attacker
                    architectures (the Phase-B result)

All are vector PDFs sized for a two-column WACV page.

Usage: python scripts/make_paper_figures.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT = "paper/figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_ORACLE = "#4a4a4a"
C_BASE = "#c44e52"
C_OURS = "#2a6f97"
C_ORIG = "#8a8a8a"


# --------------------------------------------------------------------------
def fig_teaser():
    """Two-panel conceptual figure: pooling reveals what per-frame hides, and
    shuffling tells you which mechanism is responsible."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    rng = np.random.RandomState(3)

    # ---- left: per-frame noisy evidence pools into a clear silhouette ----
    H = W = 28
    yy, xx = np.mgrid[0:H, 0:W]
    truth = (((xx - 14) / 5.0) ** 2 + ((yy - 14) / 8.0) ** 2) < 1.0
    T = 8
    frames = [truth * 0.55 + rng.randn(H, W) * 0.85 for _ in range(T)]
    pooled = np.mean(frames, axis=0)

    axL.set_xlim(0, 10); axL.set_ylim(0, 4.6); axL.axis("off")
    for i in range(3):
        ax_in = axL.inset_axes([0.02 + i * 0.11, 0.52, 0.115, 0.38])
        ax_in.imshow(frames[i], cmap="magma", vmin=-1.5, vmax=1.5)
        ax_in.set_xticks([]); ax_in.set_yticks([])
    axL.text(1.55, 4.32, r"per-frame outputs $O_t$", ha="center", fontsize=8.5)
    axL.text(3.35, 2.55, r"$\cdots$", fontsize=13, ha="center", va="center")
    axL.text(1.55, 1.85, "each frame:\nno visible class",
             ha="center", fontsize=8, color=C_BASE)

    ax_pool = axL.inset_axes([0.60, 0.42, 0.26, 0.50])
    ax_pool.imshow(pooled, cmap="magma", vmin=-1.5, vmax=1.5)
    ax_pool.set_xticks([]); ax_pool.set_yticks([])
    axL.text(7.3, 4.32, r"pooled  $\bar{O}=\mathrm{mean}_t(O_t)$",
             ha="center", fontsize=8.5)
    axL.text(7.3, 1.35, "silhouette recovered", ha="center", fontsize=8,
             color=C_OURS, weight="bold")

    arr = FancyArrowPatch((4.05, 2.9), (5.75, 2.9), arrowstyle="-|>",
                          mutation_scale=13, color="#333333", lw=1.4)
    axL.add_patch(arr)
    axL.text(4.9, 3.15, "average", ha="center", fontsize=8)
    axL.set_title("(a) Pooling recovers what per-frame audits miss", pad=6)

    # ---- right: shuffle control distinguishes the two mechanisms ----
    axR.set_xlim(0, 10); axR.set_ylim(0, 4.6); axR.axis("off")
    axR.set_title("(b) Time-shuffling identifies the mechanism", pad=6)

    axR.text(0.1, 4.05, "if the leak were carried by:", fontsize=8.5)

    # bars: ordered vs shuffled under each hypothesis
    def minibars(x0, y0, ordered, shuffled, color):
        for k, v in enumerate([ordered, shuffled]):
            axR.add_patch(plt.Rectangle((x0 + k * 0.85, y0), 0.62, v,
                                        color=color, alpha=0.9 if k == 0 else 0.35))
        axR.text(x0 + 0.31, y0 - 0.30, "ord.", ha="center", fontsize=6.5)
        axR.text(x0 + 1.16, y0 - 0.30, "shuf.", ha="center", fontsize=6.5)

    axR.text(0.35, 3.20, r"$\bf{trajectory}$", fontsize=8.5, color=C_BASE)
    axR.text(0.35, 2.80, "(order)", fontsize=7.5, color=C_BASE)
    minibars(3.35, 2.72, 1.05, 0.22, C_BASE)
    axR.text(5.75, 3.15, "shuffling destroys it", fontsize=7.5, color=C_BASE,
             va="center")

    axR.text(0.35, 1.50, r"$\bf{redundancy}$", fontsize=8.5, color=C_OURS)
    axR.text(0.35, 1.10, "(repetition)", fontsize=7.5, color=C_OURS)
    minibars(3.35, 1.02, 1.05, 1.00, C_OURS)
    axR.text(5.75, 1.45, "shuffling leaves it", fontsize=7.5, color=C_OURS,
             va="center")

    axR.text(5.0, 0.18, r"measured: $\mathrm{AP}_{\rm ord}-\mathrm{AP}_{\rm shuf}"
                        r"\approx 0 \Rightarrow$ redundancy",
             ha="center", fontsize=8, style="italic")

    plt.tight_layout()
    p = f"{OUT}/fig_teaser.pdf"
    plt.savefig(p, bbox_inches="tight"); plt.close()
    print("wrote", p)


# --------------------------------------------------------------------------
def fig_frontier():
    """Privacy-utility scatter for the car class: all 12 models."""
    zoo = json.load(open("results/seg/zoo_mamba_T16_s2_car.json"))
    util = json.load(open("results/seg/utility_car.json"))

    label = {
        "oracle_FLOOR": "oracle (floor)", "original": "original",
        "bl_FT_s1_cls48": "FT", "bl_NegGrad_s1_cls48": "NegGrad",
        "bl_NegGrad+_s1_cls48": "NegGrad+", "bl_RL_s1_cls48": "RL",
        "bl_SalUn_s1_cls48": "SalUn", "bl_SSD_s1_cls48": "SSD",
        "bl_SCRUB_s1_cls48": "SCRUB", "bl_DELETE_s1_cls48": "DELETE",
        "ADVR-T_car": "PACE-T", "ADVR-P_car": "PACE",
    }
    fig, ax = plt.subplots(figsize=(3.4, 2.9))

    for k, name in label.items():
        if k not in zoo or k not in util:
            continue
        ap = zoo[k]["ordered"]["attacker_AP"]
        sd = zoo[k]["ordered"].get("attacker_AP_std", 0.0)
        rm = util[k]["retain_mIoU"]
        if k == "oracle_FLOOR":
            c, m, s = C_ORACLE, "*", 130
        elif k == "original":
            c, m, s = C_ORIG, "s", 42
        elif k.startswith("ADVR"):
            c, m, s = C_OURS, "o", 62
        else:
            c, m, s = C_BASE, "^", 40
        ax.errorbar(rm, ap, yerr=sd, fmt="none", ecolor=c, alpha=0.45,
                    elinewidth=1.0, capsize=2)
        ax.scatter(rm, ap, c=c, marker=m, s=s, zorder=3,
                   edgecolors="white", linewidths=0.6)
        dx, dy = 0.9, 1.4
        if name == "SSD":
            dx, dy = 0.7, 2.0
        if name == "NegGrad":
            dx, dy = 0.7, -4.2
        if name == "NegGrad+":
            dx, dy = 0.8, -4.4
        if name == "PACE":
            dx, dy = -1.2, -5.0
        if name == "PACE-T":
            dx, dy = -7.0, 3.0
        if name == "oracle (floor)":
            dx, dy = -3.2, 4.0
        ax.annotate(name, (rm, ap), xytext=(rm + dx, ap + dy), fontsize=7,
                    color=c)

    ax.axhline(zoo["oracle_FLOOR"]["ordered"]["attacker_AP"], ls=":",
               color=C_ORACLE, lw=1.0)
    ax.text(1.0, zoo["oracle_FLOOR"]["ordered"]["attacker_AP"] + 1.8,
            "oracle floor", fontsize=7, color=C_ORACLE, ha="left")

    ax.set_xlabel(r"retain mIoU $\uparrow$  (utility preserved)")
    ax.set_ylabel(r"pooled attacker AP $\downarrow$  (leakage)")
    ax.set_title("Car class: privacy vs. utility", fontsize=9)
    ax.annotate("better", xy=(0.86, 0.10), xytext=(0.55, 0.30),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=7.5, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))
    plt.tight_layout()
    p = f"{OUT}/fig_frontier.pdf"
    plt.savefig(p, bbox_inches="tight"); plt.close()
    print("wrote", p)


# --------------------------------------------------------------------------
def fig_redundancy():
    """Delta_redundancy per model across three attacker architectures."""
    models = ["oracle\n(floor)", "original", "PACE-F", "PACE-T"]
    conv3d = [-0.99, 2.09, 12.69, 2.53]
    convgru = [2.02, 3.24, 19.21, 5.43]
    mamba = [5.49, 1.63, 17.49, 7.02]

    x = np.arange(len(models)); w = 0.26
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.bar(x - w, conv3d, w, label="conv3d", color="#b8c4d9", edgecolor="#4a4a4a", lw=0.5)
    ax.bar(x, convgru, w, label="ConvGRU", color="#7f9dc0", edgecolor="#4a4a4a", lw=0.5)
    ax.bar(x + w, mamba, w, label="Mamba", color=C_OURS, edgecolor="#4a4a4a", lw=0.5)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel(r"$\Delta_{\rm redundancy}$ (AP)")
    ax.set_title("Pooling gain over per-frame attack", fontsize=9)
    ax.legend(frameon=False, ncol=3, fontsize=7.5, loc="upper left")
    ax.set_ylim(-3, 24)
    plt.tight_layout()
    p = f"{OUT}/fig_redundancy.pdf"
    plt.savefig(p, bbox_inches="tight"); plt.close()
    print("wrote", p)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_teaser()
    fig_frontier()
    fig_redundancy()
