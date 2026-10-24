"""Two animated explanations for the README.

GIF 1  How one revealed label moves the prediction, and why the official
       reference blows up at n=1 while shrinkage does not.
GIF 2  What the conditional extension does: evidence from other subjects on the
       same benchmark shifts the prior before any of your own labels arrive.

Both are DIDACTIC and use synthetic illustrative numbers, which is stated on the
frames themselves. The measured results live in the figures, not here.

    python build/make_gifs.py --out docs/figures
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

INK = "#14213d"
MUTED = "#8d99ae"
ACCENT = "#1b6ca8"
WARN = "#c1121f"
GOOD = "#2a9d8f"
GRID = "#e4e7ec"
P0, S = 0.5, 3.0


def style():
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 100, "font.family": "DejaVu Sans",
        "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.facecolor": "white",
    })


def save_gif(frames, path, fps=4, loop=0, durations=None):
    """`durations` en ms por fotograma; duplicar fotogramas identicos no sirve
    porque PIL los colapsa, hay que alargar su duracion."""
    kw = {"duration": durations if durations else int(1000 / fps)}
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   loop=loop, optimize=True, **kw)
    return path


def render(fig):
    fig.canvas.draw()
    img = Image.frombytes("RGBA", fig.canvas.get_width_height(),
                          fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return img.quantize(colors=128, method=Image.MEDIANCUT).convert("P")


def gif_shrinkage(out):
    """Labels arrive one at a time; raw mean lurches, shrinkage settles."""
    rng = np.random.default_rng(11)
    truth = 0.72
    labels = (rng.random(31) < truth).astype(int)
    frames = []
    for n in range(0, 32):
        k = int(labels[:n].sum())
        raw = (k / n) if n else 0.5
        shr = (P0 * S + k) / (S + n)
        fig, (ax, bx) = plt.subplots(
            1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})

        ns = np.arange(0, n + 1)
        raws = [0.5] + [labels[:m].sum() / m for m in range(1, n + 1)]
        shrs = [(P0 * S + labels[:m].sum()) / (S + m) for m in range(0, n + 1)]
        ax.axhline(truth, color=MUTED, ls=":", lw=1.4)
        ax.text(31, truth + 0.018, "true rate of this pair", color=MUTED,
                fontsize=8.5, ha="right")
        ax.plot(ns, raws, "-o", color=WARN, lw=2.0, ms=3.5,
                label="raw mean (official reference)")
        ax.plot(ns, shrs, "-o", color=ACCENT, lw=2.6, ms=3.5,
                label="Beta-Binomial, S=3 (submitted)")
        ax.set_xlim(-0.6, 31.6); ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel("labels revealed"); ax.set_ylabel("predicted P(correct)")
        ax.set_title("One label is not evidence of certainty")
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.legend(loc="lower right", fontsize=8.5)

        bx.barh([1, 0], [raw, shr], color=[WARN, ACCENT], height=0.5)
        bx.set_yticks([1, 0]); bx.set_yticklabels(["raw mean", "shrinkage"], fontsize=10)
        bx.set_xlim(0, 1); bx.set_xlabel("prediction now")
        bx.set_title(f"n = {n}    successes = {k}")
        for y, v in ((1, raw), (0, shr)):
            bx.text(min(v + 0.03, 0.9), y, f"{v:.3f}", va="center",
                    fontsize=11, fontweight="bold",
                    color=WARN if y else ACCENT)
        if n == 1:
            bx.text(0.5, -0.78, "a single wrong label costs\nthe maximum Brier loss",
                    ha="center", fontsize=9.5, color=WARN, fontweight="bold")
        bx.grid(axis="x", color=GRID, lw=0.8); bx.set_axisbelow(True)
        fig.text(0.01, 0.012, "Synthetic illustration. Measured results are in the figures above.",
                 fontsize=7.5, color=MUTED)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        frames.append(render(fig))
    dur = [1100 if n in (0, 1, 3) else (2600 if n == 31 else 200) for n in range(32)]
    return save_gif(frames, os.path.join(out, "gif1_shrinkage.gif"), durations=dur)


def gif_peers(out):
    """Peer labels from the same benchmark move the prior before you see yours."""
    peers = [("subject A", 0.80, 20), ("subject B", 0.74, 20), ("subject C", 0.66, 20)]
    frames = []
    steps = [0, 1, 2, 3]
    for t, used in enumerate(steps):
        k = sum(int(round(p * n)) for _, p, n in peers[:used])
        tot = sum(n for _, _, n in peers[:used])
        peer_rate = (k / tot) if tot else P0
        prior = (1 - 0.5) * P0 + 0.5 * peer_rate if used else P0
        fig, ax = plt.subplots(figsize=(9.6, 4.2))
        ax.set_position([0, 0, 1, 1])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.5, 0.95, "Same benchmark, four subjects, none of your labels yet",
                ha="center", fontsize=12.5, fontweight="bold", color=INK)
        for i, (name, rate, n) in enumerate(peers):
            x0 = 0.06 + i * 0.235
            on = i < used
            col = GOOD if on else "#dfe3e8"
            ax.add_patch(plt.Rectangle((x0, 0.50), 0.20, 0.28, color=col, alpha=0.9))
            ax.text(x0 + 0.10, 0.70, name, ha="center", fontsize=10,
                    color="white" if on else MUTED, fontweight="bold")
            ax.text(x0 + 0.10, 0.60, f"{int(round(rate*n))}/{n} correct" if on else "not revealed",
                    ha="center", fontsize=9.5, color="white" if on else MUTED)
            if on:
                ax.annotate("", xy=(0.62, 0.40), xytext=(x0 + 0.10, 0.49),
                            arrowprops=dict(arrowstyle="->", color=GOOD, lw=1.6))
        ax.add_patch(plt.Rectangle((0.74, 0.50), 0.20, 0.28, color=ACCENT, alpha=0.9))
        ax.text(0.84, 0.70, "you", ha="center", fontsize=10, color="white", fontweight="bold")
        ax.text(0.84, 0.60, "0 labels", ha="center", fontsize=9.5, color="white")

        ax.plot([0.08, 0.92], [0.26, 0.26], color=GRID, lw=8, solid_capstyle="round")
        xp = 0.08 + 0.84 * prior
        ax.plot([xp], [0.26], "o", color=GOOD if used else MUTED, markersize=16,
                markeredgecolor="white", markeredgewidth=2)
        ax.text(xp, 0.18, f"prior = {prior:.3f}", ha="center", fontsize=11,
                fontweight="bold", color=GOOD if used else MUTED)
        ax.text(0.08, 0.31, "0.0", fontsize=8.5, color=MUTED)
        ax.text(0.92, 0.31, "1.0", fontsize=8.5, color=MUTED, ha="right")
        x0 = 0.08 + 0.84 * P0
        ax.plot([x0], [0.26], "|", color=MUTED, markersize=18, markeredgewidth=2)
        ax.text(x0, 0.335, "0.5 without peers", ha="center", fontsize=8.5, color=MUTED)
        msg = ("With no peer labels the prediction is exactly the submitted one"
               if used == 0 else
               f"{used} peer{'s' if used > 1 else ''} shift the prior before any of your labels")
        ax.text(0.5, 0.055, msg, ha="center", fontsize=10.5, color=INK)
        fig.text(0.01, 0.012,
                 "Synthetic illustration. Conditional on the platform supplying peer labels, "
                 "which is UNVERIFIED.", fontsize=7.5, color=MUTED)
        frames.append(render(fig))
    dur = [1500, 1200, 1200, 2600]
    return save_gif(frames[:4], os.path.join(out, "gif2_peers.gif"), durations=dur)


def still(out):
    """Static fallback image for the shrinkage GIF."""
    rng = np.random.default_rng(11)
    labels = (rng.random(31) < 0.72).astype(int)
    ns = np.arange(0, 32)
    raws = [0.5] + [labels[:m].sum() / m for m in range(1, 32)]
    shrs = [(P0 * S + labels[:m].sum()) / (S + m) for m in range(0, 32)]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.axhline(0.72, color=MUTED, ls=":", lw=1.4)
    ax.plot(ns, raws, "-o", color=WARN, lw=2.0, ms=3.5, label="raw mean (official reference)")
    ax.plot(ns, shrs, "-o", color=ACCENT, lw=2.6, ms=3.5, label="Beta-Binomial, S=3")
    ax.set_xlabel("labels revealed"); ax.set_ylabel("predicted P(correct)")
    ax.set_title("Static version of the shrinkage animation")
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); ax.legend(fontsize=9)
    fig.text(0.01, 0.01, "Synthetic illustration.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = os.path.join(out, "gif1_shrinkage_still.png")
    fig.savefig(p); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    style()
    for fn in (gif_shrinkage, gif_peers, still):
        p = fn(a.out)
        print(f"escrito: {p}  ({os.path.getsize(p)//1024} KB)")


if __name__ == "__main__":
    main()
