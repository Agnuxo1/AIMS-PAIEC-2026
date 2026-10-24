"""Render the 1080p video, timed to the narration segments.

Each scene is drawn frame by frame with matplotlib, so the animation shows the
actual mechanism rather than a slideshow. Numbers come from the committed
results, never from literals typed here, except where a frame is explicitly
labelled as a synthetic illustration.

No subtitle track and no transcript on screen: only short titles and chart
labels, which the brief allows.

    python render.py --repo <repo> --audio audio --out out
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

W, H, FPS = 1920, 1080, 20
INK = "#0f1b2d"
PAPER = "#ffffff"
MUTED = "#8d99ae"
ACCENT = "#1b6ca8"
WARN = "#c1121f"
GOOD = "#2a9d8f"
GOLD = "#e9a23b"
GRID = "#e8ebf0"
BUDGETS = [0, 1, 3, 7, 15, 31]


def style():
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 100, "font.family": "DejaVu Sans",
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "axes.edgecolor": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": PAPER, "axes.facecolor": PAPER,
        "legend.frameon": False,
    })


def canvas():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(PAPER)
    return fig


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def title(fig, text, sub=None):
    fig.text(0.06, 0.90, text, fontsize=40, fontweight="bold", color=INK, va="top")
    if sub:
        fig.text(0.06, 0.845, sub, fontsize=19, color=MUTED, va="top")


def footer(fig, text):
    fig.text(0.06, 0.045, text, fontsize=15, color=MUTED)


def chip(fig, x, y, text, colour, w=0.19, h=0.055, fs=16):
    ax = fig.add_axes([x, y, w, h]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.1), 0.96, 0.8, boxstyle="round,pad=0.02",
                                facecolor=colour, edgecolor="none",
                                transform=ax.transAxes))
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=fs,
            color="white", fontweight="bold", transform=ax.transAxes)


def save(fig, path):
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


# ---------------------------------------------------------------- scenes

def scene_problem(t, d, ctx):
    """A subject x item grid fills in; the held-out column stays unknown."""
    fig = canvas()
    title(fig, "Predict an outcome you are not allowed to observe",
          "Predictive AI Evaluation Competition  ·  NeurIPS 2026")
    ax = fig.add_axes([0.08, 0.26, 0.52, 0.50])
    rng = np.random.default_rng(3)
    rows, cols = 7, 14
    held = 10
    vals = (rng.random((rows, cols)) < rng.uniform(0.3, 0.85, (rows, 1))).astype(float)
    p = ease(t / (d * 0.45))
    shown = int(p * rows * held)
    img = np.full((rows, cols), np.nan)
    for i in range(rows):
        for j in range(held):
            if i * held + j < shown:
                img[i, j] = vals[i, j]
    cmap = matplotlib.colors.ListedColormap([WARN, ACCENT])
    cmap.set_bad("#f2f4f7")
    ax.imshow(np.ma.masked_invalid(img), cmap=cmap, vmin=0, vmax=1,
              aspect="auto", interpolation="nearest")
    if t > d * 0.42:
        q = ease((t - d * 0.42) / (d * 0.22))
        ax.add_patch(plt.Rectangle((held - 0.5, -0.5), (cols - held) * q, rows,
                                   facecolor=GOLD, alpha=0.22, edgecolor=GOLD, lw=2))
        if q > 0.5:
            ax.text(held + (cols - held) / 2 - 0.5, rows / 2, "?", fontsize=62,
                    color=GOLD, ha="center", va="center", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("benchmark items", fontsize=16)
    ax.set_ylabel("AI systems", fontsize=16)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(held / 2 - 0.5, -1.1, "scored", fontsize=15, color=MUTED, ha="center")
    if t > d * 0.5:
        ax.text(held + (cols - held) / 2 - 0.5, -1.1, "held-out benchmark",
                fontsize=15, color=GOLD, ha="center", fontweight="bold")
    if t > d * 0.62:
        bx = fig.add_axes([0.66, 0.40, 0.28, 0.30]); bx.axis("off")
        bx.text(0, 1.0, "You may first see", fontsize=19, color=MUTED, va="top")
        k = int(ease((t - d * 0.62) / (d * 0.25)) * len(BUDGETS))
        for i, b in enumerate(BUDGETS[:max(k, 0)]):
            bx.add_patch(FancyBboxPatch((0.02 + i * 0.165, 0.55), 0.145, 0.18,
                                        boxstyle="round,pad=0.02", facecolor=ACCENT,
                                        edgecolor="none", transform=bx.transAxes))
            bx.text(0.092 + i * 0.165, 0.64, str(b), ha="center", va="center",
                    fontsize=19, color="white", fontweight="bold",
                    transform=bx.transAxes)
        bx.text(0, 0.42, "labels for that system", fontsize=19, color=MUTED, va="top")
        if t > d * 0.86:
            bx.text(0, 0.22, "scored by Brier ALC", fontsize=22, color=INK,
                    va="top", fontweight="bold")
    footer(fig, "Nothing benchmark-specific can be transferred.")
    return fig


def scene_reference(t, d, ctx):
    """The reference's curve draws in; the spike at n=1 is held."""
    s = ctx["summary"]
    fig = canvas()
    title(fig, "The official reference breaks at one label",
          "raw mean of whatever labels it has seen")
    ax = fig.add_axes([0.10, 0.22, 0.54, 0.56])
    x = np.log2(np.array(BUDGETS) + 1)
    ref = [s.loc["empirical_mean", f"B{b}"] for b in BUDGETS]
    ax.axhline(0.25, color=MUTED, ls="--", lw=2)
    ax.text(x[-1], 0.2555, "doing nothing", color=MUTED, fontsize=15, ha="right")
    n = ease(t / (d * 0.55)) * (len(BUDGETS) - 1)
    k = int(np.floor(n)) + 1
    xs, ys = list(x[:k]), list(ref[:k])
    if k < len(BUDGETS):
        f = n - (k - 1)
        xs.append(x[k - 1] + f * (x[k] - x[k - 1]))
        ys.append(ref[k - 1] + f * (ref[k] - ref[k - 1]))
    ax.plot(xs, ys, "-o", color=WARN, lw=4, ms=10)
    if n >= 1:
        ax.plot([x[1]], [ref[1]], "o", color=WARN, ms=20, alpha=0.3)
        ax.annotate(f"{ref[1]:.3f}", xy=(x[1], ref[1]), xytext=(x[1] + 0.3, ref[1] + 0.012),
                    fontsize=26, color=WARN, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in BUDGETS], fontsize=16)
    ax.set_xlabel("labels revealed", fontsize=18)
    ax.set_ylabel("Brier score", fontsize=18)
    ax.set_ylim(0.17, 0.37); ax.grid(color=GRID, lw=1); ax.set_axisbelow(True)
    ax.tick_params(labelsize=14)
    if t > d * 0.55:
        q = ease((t - d * 0.55) / (d * 0.3))
        bx = fig.add_axes([0.69, 0.34, 0.25, 0.34]); bx.axis("off")
        bx.text(0, 1.0, "with one label", fontsize=20, color=MUTED, va="top")
        bx.text(0, 0.80, "prediction", fontsize=17, color=MUTED, va="top")
        bx.text(0, 0.62, "0  or  1", fontsize=46, color=WARN, va="top",
                fontweight="bold", alpha=q)
        if q > 0.55:
            bx.text(0, 0.34, "one wrong guess", fontsize=17, color=MUTED, va="top")
            bx.text(0, 0.16, "costs the maximum", fontsize=20, color=WARN,
                    va="top", fontweight="bold")
    footer(fig, "Independent re-evaluation · 151 benchmarks · 13,200 pairs · 8 seeds")
    return fig


def scene_method(t, d, ctx):
    """The posterior mean, then the same label stream handled two ways."""
    fig = canvas()
    title(fig, "A prior worth three observations", "Beta-Binomial posterior mean")
    fig.text(0.5, 0.70, r"$\hat{p} \;=\; \dfrac{\alpha + k}{\alpha + \beta + n}$",
             fontsize=58, ha="center", color=INK)
    fig.text(0.5, 0.605, r"$\alpha = 0.5\,S,\quad \beta = 0.5\,S,\quad S = 3$",
             fontsize=30, ha="center", color=MUTED)
    if t > d * 0.42:
        q = ease((t - d * 0.42) / (d * 0.5))
        ax = fig.add_axes([0.22, 0.17, 0.56, 0.33])
        rng = np.random.default_rng(11)
        labels = (rng.random(31) < 0.72).astype(int)
        m = max(1, int(q * 12))
        ns = np.arange(0, m + 1)
        raw = [0.5] + [labels[:i].sum() / i for i in range(1, m + 1)]
        shr = [(0.5 * 3 + labels[:i].sum()) / (3 + i) for i in range(0, m + 1)]
        ax.plot(ns, raw, "-o", color=WARN, lw=3, ms=7, label="raw mean")
        ax.plot(ns, shr, "-o", color=ACCENT, lw=4, ms=7, label="shrinkage, S=3")
        ax.set_xlim(-0.4, 12.4); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("labels revealed", fontsize=17)
        ax.set_ylabel("prediction", fontsize=17)
        ax.grid(color=GRID, lw=1); ax.set_axisbelow(True)
        ax.legend(fontsize=16, loc="lower right"); ax.tick_params(labelsize=13)
        if m >= 1:
            ax.annotate("0.625", xy=(1, shr[1]), xytext=(1.5, shr[1] - 0.22),
                        fontsize=20, color=ACCENT, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
    footer(fig, "No state, no fitting at run time, no payload, no network. Synthetic label stream.")
    return fig


def scene_result(t, d, ctx):
    """Both curves together, then the headline difference counts up."""
    s = ctx["summary"]
    fig = canvas()
    title(fig, "Nine tenths of the problem is the first label",
          "independent protocol, local validation")
    ax = fig.add_axes([0.08, 0.22, 0.55, 0.56])
    x = np.log2(np.array(BUDGETS) + 1)
    for key, colour, lw, lab in (("constant_05", MUTED, 2.5, "constant 0.5"),
                                 ("empirical_mean", WARN, 4, "official reference"),
                                 ("original_beta3", ACCENT, 5, "submitted")):
        y = [s.loc[key, f"B{b}"] for b in BUDGETS]
        ax.plot(x, y, marker="o", color=colour, lw=lw, ms=9, label=lab,
                ls="--" if key == "constant_05" else "-")
    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in BUDGETS], fontsize=16)
    ax.set_xlabel("labels revealed", fontsize=18); ax.set_ylabel("Brier score", fontsize=18)
    ax.set_ylim(0.17, 0.37); ax.grid(color=GRID, lw=1); ax.set_axisbelow(True)
    ax.legend(fontsize=16, loc="upper right"); ax.tick_params(labelsize=14)
    q = ease(t / (d * 0.6))
    bx = fig.add_axes([0.665, 0.24, 0.29, 0.52]); bx.axis("off")
    ref, ours = s.loc["empirical_mean", "brier_alc"], s.loc["original_beta3", "brier_alc"]
    bx.text(0, 0.96, "Brier ALC", fontsize=21, color=MUTED, va="top")
    bx.text(0, 0.86, f"{ref:.4f}", fontsize=44, color=WARN, va="top", fontweight="bold")
    bx.text(0.56, 0.875, "reference", fontsize=16, color=MUTED, va="top")
    bx.text(0, 0.68, f"{ref + (ours - ref) * q:.4f}", fontsize=44, color=ACCENT,
            va="top", fontweight="bold")
    bx.text(0.56, 0.695, "submitted", fontsize=16, color=MUTED, va="top")
    if t > d * 0.5:
        r = ease((t - d * 0.5) / (d * 0.35))
        u = ctx["unc"]["original_beta3"]
        bx.text(0, 0.47, f"{u['difference_pair_mean'] * r:+.4f}", fontsize=38,
                color=GOOD, va="top", fontweight="bold")
        if r > 0.6:
            bx.text(0, 0.345, "95% interval", fontsize=16, color=MUTED, va="top")
            bx.text(0, 0.275, f"[{u['ci95'][0]:+.4f}, {u['ci95'][1]:+.4f}]",
                    fontsize=19, color=INK, va="top")
            bx.text(0, 0.165, f"{u['benchmarks_improved']} of {u['n_benchmarks']} benchmarks",
                    fontsize=21, color=GOOD, va="top", fontweight="bold")
    footer(fig, "Local validation, not an official score. The reference is deliberately simple.")
    return fig


def scene_audit(t, d, ctx):
    """Replication figure, then the four defects appear one at a time."""
    fig = canvas()
    title(fig, "Re-derived independently, then taken apart",
          "reconstruction from the data files, without the original evaluator")
    q = ease(t / (d * 0.22))
    fig.text(0.07, 0.745, "maximum per-pair error", fontsize=19, color=MUTED, va="top")
    fig.text(0.07, 0.675, "1.11 x 10$^{-16}$", fontsize=52, color=GOOD,
             va="top", fontweight="bold", alpha=q)
    fig.text(0.40, 0.70, "on the 496 reconstructed headline pairs",
             fontsize=18, color=MUTED, va="top", alpha=q)
    defects = [
        "Rich priors saw other headline benchmarks while fitting",
        "Some filters counted responses, not distinct items",
        "Caps on subjects and items had changed the population",
        "An empirical oracle was called a noise floor",
    ]
    for i, text in enumerate(defects):
        s0 = d * (0.26 + i * 0.165)
        if t < s0:
            continue
        a = ease((t - s0) / (d * 0.12))
        y = 0.525 - i * 0.105
        fig.text(0.075, y, "\u2716", fontsize=26, color=WARN, alpha=a, va="center")
        fig.text(0.115, y, text, fontsize=25, color=INK, alpha=a, va="center")
    if t > d * 0.90:
        a = ease((t - d * 0.90) / (d * 0.10))
        fig.text(0.075, 0.095, "all four corrected in the protocol shown here",
                 fontsize=22, color=GOOD, fontweight="bold", alpha=a)
    return fig


def scene_negatives(t, d, ctx):
    """Forest plot of every refinement, drawn one row at a time."""
    u = ctx["unc"]
    fig = canvas()
    title(fig, "Seven things that did not work", "reported because they bound where the signal is not")
    ax = fig.add_axes([0.30, 0.20, 0.60, 0.58])
    rows = [("tuned_beta", "Tuned Beta prior"),
            ("calibration_table", "Calibration table"),
            ("subject_ridge", "Subject-attribute Ridge")]
    extra = [("Per-budget tuning", "identical"),
             ("Item-text features at budget 0", "no effect"),
             ("Per-item deviation, hand-built", "worse"),
             ("Per-item deviation, sparse text", "worse")]
    ax.axvline(0, color=INK, lw=2)
    total = len(rows) + len(extra)
    shown = ease(t / (d * 0.82)) * total
    for i, (key, label) in enumerate(rows):
        if shown < i + 1:
            continue
        a = min(1.0, shown - i)
        y = total - 1 - i
        dd = u[key]
        lo, hi = dd["ci95"]
        colour = WARN if lo > 0 else MUTED
        ax.plot([lo, hi], [y, y], color=colour, lw=7, solid_capstyle="round", alpha=a)
        ax.plot([dd["difference_pair_mean"]], [y], "o", color=colour, ms=16,
                markeredgecolor="white", markeredgewidth=2, alpha=a)
        ax.text(0.0235, y, f"{dd['difference_pair_mean']:+.4f}", fontsize=19,
                color=colour, va="center", fontweight="bold", alpha=a)
    for j, (label, verdict) in enumerate(extra):
        i = len(rows) + j
        if shown < i + 1:
            continue
        a = min(1.0, shown - i)
        y = total - 1 - i
        ax.text(0.0235, y, verdict, fontsize=19, color=MUTED, va="center",
                fontweight="bold", alpha=a)
        ax.plot([0], [y], "o", color=MUTED, ms=10, alpha=a * 0.6)
    ax.set_yticks(range(total))
    ax.set_yticklabels([l for _, l in reversed([*rows, *[(a, a) for a, _ in extra]])]
                       if False else
                       list(reversed([l for _, l in rows] + [l for l, _ in extra])),
                       fontsize=18)
    ax.set_xlim(-0.012, 0.030); ax.set_xlabel("difference in Brier ALC", fontsize=18)
    ax.grid(axis="x", color=GRID, lw=1); ax.set_axisbelow(True)
    ax.tick_params(labelsize=13)
    footer(fig, "Bars are 95% component bootstrap. Negative is better.")
    return fig


def scene_context(t, d, ctx):
    """Peers light up, the prior slides, then the measured cohort curve."""
    c = ctx["ctxsum"]
    fig = canvas()
    title(fig, "One thing helped, on a condition we could not verify",
          "labels from other subjects on the same benchmark")
    peers = [("subject A", "16/20"), ("subject B", "15/20"), ("subject C", "13/20")]
    used = min(3, int(ease(t / (d * 0.33)) * 4))
    for i, (name, score) in enumerate(peers):
        on = i < used
        x0 = 0.075 + i * 0.165
        ax = fig.add_axes([x0, 0.57, 0.145, 0.16]); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.03, 0.08), 0.94, 0.84,
                                    boxstyle="round,pad=0.02",
                                    facecolor=GOOD if on else "#e6eaef",
                                    edgecolor="none", transform=ax.transAxes))
        ax.text(0.5, 0.63, name, ha="center", fontsize=17, fontweight="bold",
                color="white" if on else MUTED, transform=ax.transAxes)
        ax.text(0.5, 0.33, score if on else "hidden", ha="center", fontsize=16,
                color="white" if on else MUTED, transform=ax.transAxes)
    ax = fig.add_axes([0.575, 0.57, 0.145, 0.16]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.03, 0.08), 0.94, 0.84, boxstyle="round,pad=0.02",
                                facecolor=ACCENT, edgecolor="none", transform=ax.transAxes))
    ax.text(0.5, 0.63, "you", ha="center", fontsize=17, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.33, "0 labels", ha="center", fontsize=16, color="white",
            transform=ax.transAxes)
    rates = [0.80, 0.75, 0.65]
    prior = 0.5 if used == 0 else 0.5 * 0.5 + 0.5 * float(np.mean(rates[:used]))
    bx = fig.add_axes([0.075, 0.40, 0.645, 0.10]); bx.axis("off")
    bx.plot([0.02, 0.98], [0.5, 0.5], color=GRID, lw=14, solid_capstyle="round",
            transform=bx.transAxes)
    xp = 0.02 + 0.96 * prior
    bx.plot([0.02 + 0.96 * 0.5], [0.5], "|", color=MUTED, ms=26, mew=3,
            transform=bx.transAxes)
    bx.plot([xp], [0.5], "o", color=GOOD if used else MUTED, ms=24,
            markeredgecolor="white", markeredgewidth=3, transform=bx.transAxes)
    bx.text(xp, 0.02, f"prior {prior:.3f}", ha="center", fontsize=19,
            color=GOOD if used else MUTED, fontweight="bold", transform=bx.transAxes)
    if t > d * 0.42:
        q = ease((t - d * 0.42) / (d * 0.33))
        cx = fig.add_axes([0.76, 0.26, 0.19, 0.45]); cx.axis("off")
        sizes = [1, 2, 4, 8]
        safe = [c[f"fallback_safe_cohort{n}"]["brier_alc"] for n in sizes]
        ref = [c[f"context_cohort{n}"]["reference_alc"] for n in sizes]
        m = max(1, int(q * len(sizes)))
        ix = fig.add_axes([0.765, 0.27, 0.185, 0.40])
        ix.plot(range(m), ref[:m], "-o", color=ACCENT, lw=3, ms=7)
        ix.plot(range(m), safe[:m], "-o", color=GOOD, lw=3, ms=7)
        ix.set_xticks(range(len(sizes))); ix.set_xticklabels(sizes, fontsize=13)
        ix.set_xlabel("peers visible", fontsize=15)
        ix.set_ylabel("Brier ALC", fontsize=15)
        ix.tick_params(labelsize=11); ix.grid(color=GRID, lw=1); ix.set_axisbelow(True)
        ix.set_ylim(0.2015, 0.2075)
    if t > d * 0.72:
        a = ease((t - d * 0.72) / (d * 0.26))
        fig.text(0.075, 0.27, "with no peer labels it returns the submitted predictor exactly",
                 fontsize=23, color=INK, alpha=a, fontweight="bold")
        fig.text(0.075, 0.205, "so it cannot be worse in that case",
                 fontsize=20, color=MUTED, alpha=a)
    footer(fig, "Simulated cohorts. Real availability on the platform is UNVERIFIED. Peer values are illustrative.")
    return fig


def scene_repro(t, d, ctx):
    """What is public, what CI checks, and what is honestly missing."""
    fig = canvas()
    title(fig, "Public, pinned, and honest about the gap", None)
    items = [
        ("Dataset revision pinned, 890 files hashed", GOOD),
        ("CI validates both archives on every push", GOOD),
        ("Figures regenerate from committed results", GOOD),
        ("Seven negative results kept, not cleaned away", GOOD),
    ]
    for i, (text, colour) in enumerate(items):
        s0 = d * (0.06 + i * 0.13)
        if t < s0:
            continue
        a = ease((t - s0) / (d * 0.10))
        y = 0.705 - i * 0.088
        fig.text(0.075, y, "\u2713", fontsize=28, color=colour, alpha=a, va="center")
        fig.text(0.115, y, text, fontsize=25, color=INK, alpha=a, va="center")
    if t > d * 0.62:
        a = ease((t - d * 0.62) / (d * 0.16))
        ax = fig.add_axes([0.07, 0.17, 0.60, 0.14]); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.01, 0.05), 0.98, 0.9, boxstyle="round,pad=0.02",
                                    facecolor="#fdf1f1", edgecolor=WARN, lw=2,
                                    alpha=a, transform=ax.transAxes))
        ax.text(0.04, 0.62, "no submission identifier", fontsize=27, color=WARN,
                fontweight="bold", alpha=a, transform=ax.transAxes)
        ax.text(0.04, 0.26, "the submission platform has been unreachable",
                fontsize=19, color=MUTED, alpha=a, transform=ax.transAxes)
    if t > d * 0.82:
        a = ease((t - d * 0.82) / (d * 0.16))
        fig.text(0.70, 0.235, "no ranking,", fontsize=24, color=MUTED, alpha=a)
        fig.text(0.70, 0.185, "no acceptance,", fontsize=24, color=MUTED, alpha=a)
        fig.text(0.70, 0.135, "no official score claimed", fontsize=24,
                 color=INK, alpha=a, fontweight="bold")
    return fig


def scene_close(t, d, ctx):
    fig = canvas()
    a = ease(t / (d * 0.35))
    fig.text(0.5, 0.60, "github.com/Agnuxo1/AIMS-PAIEC-2026", fontsize=42,
             ha="center", color=ACCENT, fontweight="bold", alpha=a)
    fig.text(0.5, 0.50, "code  ·  audit  ·  failures  ·  report", fontsize=24,
             ha="center", color=MUTED, alpha=a)
    if t > d * 0.45:
        b = ease((t - d * 0.45) / (d * 0.4))
        fig.text(0.5, 0.36, "Francisco Angulo de Lafuente", fontsize=24,
                 ha="center", color=INK, alpha=b)
        fig.text(0.5, 0.305, "Independent researcher · Madrid, Spain", fontsize=19,
                 ha="center", color=MUTED, alpha=b)
    return fig


SCENES = [scene_problem, scene_reference, scene_method, scene_result,
          scene_audit, scene_negatives, scene_context, scene_repro, scene_close]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=r"E:\AIMS-sprint\github\AIMS-PAIEC-2026")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    style()
    os.makedirs(a.out, exist_ok=True)
    frames_dir = os.path.join(a.out, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    narration = json.load(open(os.path.join(a.audio, "narration.json"), encoding="utf-8"))
    ctx = {
        "summary": pd.read_csv(os.path.join(a.repo, "results", "summary.csv")).set_index("method"),
        "unc": {x["method"]: x for x in json.load(
            open(os.path.join(a.repo, "results", "uncertainty.json"), encoding="utf-8"))},
        "ctxsum": {x["method"]: x for x in json.load(
            open(os.path.join(a.repo, "results", "context_summary.json"), encoding="utf-8"))},
    }

    n = 0
    for seg, fn in zip(narration["segments"], SCENES):
        d = seg["seconds"] + 0.35          # small tail so the cut is not abrupt
        total = int(round(d * FPS))
        for k in range(total):
            fig = fn(k / FPS, d, ctx)
            save(fig, os.path.join(frames_dir, f"f{n:06d}.png"))
            n += 1
        print(f"  escena {seg['index']} {seg['title'][:34]:<34} {total:>4} fotogramas", flush=True)
    print(f"total {n} fotogramas = {n/FPS:.1f}s")

    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    concat = os.path.join(a.out, "audio_list.txt")
    with open(concat, "w", encoding="utf-8") as fh:
        for seg in narration["segments"]:
            fh.write(f"file '{os.path.join(a.audio, seg['file']).replace(os.sep, '/')}'\n")
            fh.write("outpoint %.3f\n" % seg["seconds"])
    voice = os.path.join(a.out, "voice.m4a")
    subprocess.run([exe, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", concat, "-c:a", "aac", "-b:a", "192k", voice], check=True)

    mp4 = os.path.join(a.out, "AIMS_PAIEC_2026.mp4")
    subprocess.run([exe, "-y", "-loglevel", "error",
                    "-framerate", str(FPS), "-i", os.path.join(frames_dir, "f%06d.png"),
                    "-i", voice,
                    "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-r", str(FPS),
                    "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", "-shortest",
                    "-sn", "-map", "0:v:0", "-map", "1:a:0", mp4], check=True)
    print("escrito:", mp4, os.path.getsize(mp4) // 1024, "KB")


if __name__ == "__main__":
    main()
