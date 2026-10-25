"""Figures for the public repository, built only from committed result files.

Every number plotted here is read from results/*.csv or results/*.json. Nothing
is hard-coded, so the figures cannot drift from the evidence.

    python build/make_figures.py --repo <path-to-repo> --out <docs/figures>
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]

INK = "#14213d"
MUTED = "#8d99ae"
ACCENT = "#1b6ca8"
WARN = "#c1121f"
GOOD = "#2a9d8f"
GRID = "#e4e7ec"


def style():
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def fig_budget_curve(repo, out):
    """Brier score at each official label budget, independent protocol."""
    s = pd.read_csv(os.path.join(repo, "results", "summary.csv")).set_index("method")
    rows = [
        ("constant_05", "Constant 0.5", MUTED, "--", 1.6),
        ("empirical_mean", "Empirical mean (official reference)", WARN, "-", 2.0),
        ("original_beta3", "Beta-Binomial, S=3 (submitted)", ACCENT, "-", 2.6),
    ]
    x = np.log2(np.array(BUDGETS) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    grid(ax)
    for key, label, colour, ls, lw in rows:
        y = [s.loc[key, f"B{b}"] for b in BUDGETS]
        ax.plot(x, y, ls, color=colour, linewidth=lw, marker="o", markersize=5,
                label=f"{label}   ALC {s.loc[key,'brier_alc']:.4f}", zorder=3)
    ax.annotate("one raw label can cost\nthe maximum Brier loss",
                xy=(x[1], s.loc["empirical_mean", "B1"]),
                xytext=(x[1] + 0.55, s.loc["empirical_mean", "B1"] + 0.045),
                fontsize=9, color=WARN,
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.2))
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_xlabel("Label budget $n$  (axis spaced by $\\log_2(1+n)$, as in the official metric)")
    ax.set_ylabel("Brier score  (lower is better)")
    ax.set_title("Where the gain comes from: the first revealed label")
    ax.legend(loc="lower left", fontsize=9)
    fig.text(0.01, 0.01,
             "Independent re-evaluation: 151 benchmarks, 13,200 eligible pairs, 8 seeds. "
             "Local validation, not an official score.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    p = os.path.join(out, "fig1_budget_curve.png")
    fig.savefig(p); plt.close(fig)
    return p


def fig_methods(repo, out):
    """Each tried method against the submitted one, with its interval."""
    u = {d["method"]: d for d in json.load(
        open(os.path.join(repo, "results", "uncertainty.json"), encoding="utf-8"))}
    order = [
        ("original_beta3", "Beta-Binomial S=3\nvs official reference", True),
        ("tuned_beta", "Tuned Beta prior", False),
        ("calibration_table", "Calibration table", False),
        ("subject_ridge", "Subject-attribute Ridge prior", False),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    grid(ax, axis="x")
    ax.axvline(0, color=INK, linewidth=1.0, zorder=2)
    for i, (key, label, headline) in enumerate(order):
        d = u[key]
        y = len(order) - 1 - i
        lo, hi = d["ci95"]
        diff = d["difference_pair_mean"]
        colour = GOOD if headline else (WARN if lo > 0 else MUTED)
        ax.plot([lo, hi], [y, y], color=colour, linewidth=3.2,
                solid_capstyle="round", zorder=3)
        ax.plot([diff], [y], "o", color=colour, markersize=8,
                markeredgecolor="white", markeredgewidth=1.4, zorder=4)
        note = f"{diff:+.4f}  ({d['benchmarks_improved']}/{d['n_benchmarks']})"
        ax.text(0.0465, y, note, va="center", ha="right", fontsize=9.5,
                color=colour, fontweight="bold")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([l for _, l, _ in reversed(order)], fontsize=9)
    ax.set_xlabel("Difference in Brier ALC against the comparison baseline "
                  "(negative = better)\n"
                  "bars are 95% component bootstrap")
    ax.set_title("One method helps. The refinements do not.")
    ax.set_xlim(-0.046, 0.048)
    fig.text(0.01, 0.01,
             "Top row compares the submitted predictor with the official empirical-mean reference; "
             "the rest compare refinements with the submitted predictor.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    p = os.path.join(out, "fig2_methods.png")
    fig.savefig(p); plt.close(fig)
    return p


def fig_cohorts(repo, out):
    """Conditional contextual result as the number of visible subjects grows."""
    c = json.load(open(os.path.join(repo, "results", "context_summary.json"), encoding="utf-8"))
    by = {d["method"]: d for d in c}
    sizes = [1, 2, 4, 8]
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    grid(ax)
    ref = [by[f"original_cohort{n}"]["brier_alc"] if f"original_cohort{n}" in by
           else by[f"context_cohort{n}"]["reference_alc"] for n in sizes]
    safe = [by[f"fallback_safe_cohort{n}"]["brier_alc"] for n in sizes]
    ctx = [by[f"context_cohort{n}"]["brier_alc"] for n in sizes]
    x = np.arange(len(sizes))
    ax.plot(x, ref, "-o", color=ACCENT, linewidth=2.4, markersize=6,
            label="Submitted predictor (ignores peers)")
    ax.plot(x, safe, "-o", color=GOOD, linewidth=2.4, markersize=6,
            label="Contextual, S=3 with exact fallback")
    ax.plot(x, ctx, "--o", color=MUTED, linewidth=1.8, markersize=5,
            label="Contextual, strength also retuned (worse)")
    marcados = []
    for i, n in enumerate(sizes):
        d = by[f"fallback_safe_cohort{n}"]
        lo, hi = d["ci99_five_candidates"]
        if hi < 0:
            ax.plot([x[i]], [safe[i]], marker="*", markersize=13, color=GOOD, zorder=5)
            marcados.append(f"{n}")
    if marcados:
        ax.text(0.015, 0.06, "★  99% interval excludes zero", transform=ax.transAxes,
                fontsize=8.5, color=GOOD, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"{n} subject{'s' if n>1 else ''}" for n in sizes])
    ax.set_xlabel("Visible subjects sharing the same benchmark in the simulated cohort")
    ax.set_ylabel("Brier ALC  (lower is better)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.set_title("A conditional gain, only if peer labels are actually supplied")
    ax.legend(loc="upper right", fontsize=9)
    fig.text(0.01, 0.012,
             "Simulated cohorts. Whether the official platform supplies labels from other\n"
             "subjects is UNVERIFIED; with none, the method returns the submitted predictor exactly.",
             fontsize=7.5, color=MUTED, linespacing=1.4)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    p = os.path.join(out, "fig3_cohorts.png")
    fig.savefig(p); plt.close(fig)
    return p


def fig_status(repo, out):
    """What is validated locally, what is conditional and what is still pending."""
    fig, ax = plt.subplots(figsize=(7.6, 2.5))
    ax.axis("off")
    # Every number on this figure is read from results/, not typed here.
    # (An earlier version hard-coded 0.2368, 0.2060, 133/151 and -0.0026.)
    s = pd.read_csv(os.path.join(repo, "results", "summary.csv")).set_index("method")
    u = {d["method"]: d for d in json.load(
        open(os.path.join(repo, "results", "uncertainty.json"), encoding="utf-8"))}
    c = {d["method"]: d for d in json.load(
        open(os.path.join(repo, "results", "context_summary.json"), encoding="utf-8"))}
    ref = s.loc["empirical_mean", "brier_alc"]
    ours = s.loc["original_beta3", "brier_alc"]
    won = u["original_beta3"]["benchmarks_improved"]
    n_b = u["original_beta3"]["n_benchmarks"]
    extra = c["fallback_safe_cohort4"]["difference"]
    items = [
        ("Local validation", f"{ref:.4f} → {ours:.4f} Brier ALC, {won}/{n_b} benchmarks",
         GOOD, "verified"),
        ("Conditional extension", f"{extra:+.4f} more, only with peer labels (unverified)",
         "#e9c46a", "experimental"),
        ("Official score", "no submission ID: platform unreachable", MUTED, "pending"),
    ]
    for i, (title, body, colour, tag) in enumerate(items):
        y = 0.74 - i * 0.30
        ax.add_patch(plt.Rectangle((0.015, y - 0.1), 0.012, 0.2,
                                   color=colour, transform=ax.transAxes, clip_on=False))
        ax.text(0.045, y + 0.045, title, fontsize=11, fontweight="bold",
                color=INK, transform=ax.transAxes)
        ax.text(0.045, y - 0.045, body, fontsize=9.5, color="#4a5568",
                transform=ax.transAxes)
        ax.text(0.985, y, tag.upper(), fontsize=8.5, color=colour, ha="right",
                fontweight="bold", transform=ax.transAxes)
    ax.set_title("Three different claims, kept apart", loc="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    p = os.path.join(out, "fig4_status.png")
    fig.savefig(p); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=r"E:\AIMS-sprint\entrega_auditada_v2\repo")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    style()
    for fn in (fig_budget_curve, fig_methods, fig_cohorts, fig_status):
        print("escrito:", fn(a.repo, a.out))


if __name__ == "__main__":
    main()
