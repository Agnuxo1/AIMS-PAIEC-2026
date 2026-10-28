"""Figures and animations for experiment E1 and the Codabench feedback.

fig5_e1_item_peer.png   Measured E1 result (held-out folds, 8 seeds), from experiments/E1.
gif3_item_peer.gif      DIDACTIC: how a peer's label on the same item moves one prediction.
gif4_codabench_alc.gif  MEASURED: the official formative learning curve of each Codabench
                        submission, built budget by budget into its Brier ALC.

    python build/make_e1_figures.py --repo . --out docs/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from make_gifs import ACCENT, GOOD, GRID, INK, MUTED, WARN, render, save_gif, style

BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = [.1, .2, .2, .2, .2, .1]
LABELS = {"A|0|0": "Submitted\n(E0, 1013718)", "B|0|0": "Benchmark\ncontext (CTX)", "B|0.75|1": "Same-item\npeers (E1)"}


def seed_table(path):
    d = pd.read_parquet(path)
    d["cfg"] = d.base + "|" + d.lam.map(lambda v: f"{v:g}") + "|" + d.c.map(lambda v: f"{v:g}")
    d = d[d.cfg.isin(LABELS)]
    # cohort -> benchmark -> equal-weight benchmark mean, per seed
    b = d.groupby(["seed", "benchmark_id", "cfg"]).alc.mean().reset_index()
    return b.groupby(["seed", "cfg"]).alc.mean().unstack()


def fig_e1(repo, out):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    colours = [MUTED, ACCENT, GOOD]
    for g, m in enumerate((2, 4)):
        t = seed_table(repo / f"experiments/E1/e1_scores_m{m}.parquet")
        for i, cfg in enumerate(LABELS):
            x = g * 4 + i
            ax.bar(x, t[cfg].mean(), color=colours[i], width=0.8, zorder=2,
                   label=LABELS[cfg].replace("\n", " ") if g == 0 else None)
            ax.errorbar(x, t[cfg].mean(), yerr=t[cfg].std(ddof=1), color=INK, capsize=3, lw=1, zorder=3)
            ax.text(x, t[cfg].mean() + 0.0006, f"{t[cfg].mean():.4f}", ha="center", fontsize=7.5)
    ax.set_xticks([1, 5]); ax.set_xticklabels(["2 subjects / benchmark", "4 subjects / benchmark"])
    ax.set_ylim(0.200, 0.2165); ax.set_ylabel("Brier ALC (lower is better)")
    ax.set_title("Platform-like cohorts (bars: mean, whiskers: SD over 8 seeds)", fontsize=10)
    ax.grid(axis="y", color=GRID, zorder=0); ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    rows = []
    for m in (2, 4):
        r = json.loads((repo / f"experiments/E1/e1_report_m{m}.json").read_text())
        for key, name in (("E1(B) - A|0|0", "E1 vs submitted"), ("E1(B) - B|0|0", "E1 vs CTX")):
            x = r[key]; rows.append((f"{name}\n{m} subjects", x["mean_diff"], *x["ci95"], x["seeds_negative"]))
    for i, (name, d, lo, hi, neg) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=GOOD, lw=3, solid_capstyle="round")
        ax.plot(d, i, "o", color=INK, ms=6)
        ax.text(hi + 0.0003, i, f"{d:+.4f}  ({neg}/8 seeds)", va="center", fontsize=8)
    ax.axvline(0, color=WARN, lw=1, ls="--")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(-0.0115, 0.0045)
    ax.set_xticks([-0.010, -0.005, 0.0]); ax.set_xlabel("Brier ALC difference (95% bootstrap)")
    ax.set_title("Effect sizes"); ax.grid(axis="x", color=GRID)
    fig.text(0.01, 0.01, "Simulated cohorts on public benchmarks; every pair here shares its benchmark with a peer. "
             "On Codabench only part of each sample does, so the platform gain is diluted.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "fig5_e1_item_peer.png", dpi=110)
    plt.close(fig)


def gif_item_peer(out):
    """Synthetic, didactic: subject A's evaluation items, subject B revealing labels."""
    style()
    rng = np.random.default_rng(7)
    n_items = 24
    difficulty = np.clip(rng.normal(0.55, 0.28, n_items), 0.03, 0.97)   # true item success rates
    a_eval = np.arange(0, n_items, 2)                                     # A's evaluation pool
    b_order = rng.permutation(n_items)[:9]                                # B's acquired items
    b_y = (rng.random(9) < difficulty[b_order]).astype(int)
    a_k, a_n = 2, 3                                                       # A has seen 3 own labels
    frames, durations = [], []
    for step in range(0, 10):
        fig = plt.figure(figsize=(9.6, 4.4))
        ax = fig.add_axes([0.05, 0.18, 0.62, 0.66])
        rev = b_order[:step]; ys = b_y[:step]
        kB, nB = ys.sum(), len(ys)
        base = (a_k + 3 * (0.5 + 0.5 * (((1.5 + kB) / (3 + nB)) - 0.5))) / (a_n + 3)
        preds = np.full(n_items, np.nan)
        for it in a_eval:
            hit = [(y, (kB - y + 1.5) / (nB - 1 + 3)) for i2, y in zip(rev, ys) if i2 == it]
            shift = 0.75 * sum(y - r for y, r in hit) / (1 + len(hit)) if hit else 0.0
            preds[it] = np.clip(base + shift, 1e-6, 1 - 1e-6)
        for it in range(n_items):
            x, yrow = it % 12, 1 - it // 12
            is_eval = it in a_eval
            face = "white"
            if it in rev:
                face = GOOD if b_y[list(b_order).index(it)] == 1 else WARN
            ax.add_patch(plt.Rectangle((x, yrow * 1.6), 0.9, 0.9, facecolor=face,
                                       edgecolor=ACCENT if is_eval else MUTED, lw=2.2 if is_eval else 0.8))
            if is_eval:
                moved = it in rev
                ax.text(x + 0.45, yrow * 1.6 + 0.45, f"{preds[it]:.2f}", ha="center", va="center",
                        fontsize=8.5, fontweight="bold" if moved else "normal", color=INK)
        ax.set_xlim(-0.2, 12.2); ax.set_ylim(-0.3, 2.8); ax.axis("off")
        ax.set_title(f"Subject B has revealed {step} label{'s' if step != 1 else ''} on the same benchmark", loc="left")
        side = fig.add_axes([0.70, 0.18, 0.28, 0.66]); side.axis("off")
        side.text(0, 0.95, "Blue frame  = item A will be scored on", fontsize=8.5, color=ACCENT)
        side.text(0, 0.86, "Green / red = B's revealed success / failure", fontsize=8.5)
        side.text(0, 0.70, f"A's own labels: {a_k}/{a_n}", fontsize=9)
        side.text(0, 0.61, f"B's labels so far: {kB}/{nB}", fontsize=9)
        side.text(0, 0.50, f"Pair-level base for A: {base:.3f}", fontsize=9, fontweight="bold")
        side.text(0, 0.02, "When B reveals an item A will be\nscored on, A's prediction for that\n"
                           "item moves by 0.75 x B's residual\n(label minus B's own rate).", fontsize=8.5)
        fig.text(0.05, 0.92, "Same-item peer evidence (E1)", fontsize=13, fontweight="bold")
        fig.text(0.05, 0.06, "Illustration with synthetic numbers. Measured effects: fig5_e1_item_peer.png.",
                 fontsize=8, color=MUTED)
        frames.append(render(fig)); durations.append(1800 if step in (0, 9) else 900)
    save_gif(frames, out / "gif3_item_peer.gif", durations=durations)


def gif_codabench(repo, out):
    """Measured formative feedback of every Codabench submission saved in docs/evidence."""
    style()
    curves = {}
    for path in sorted((repo / "docs/evidence").glob("codabench_*_formative.csv")):
        sid = path.stem.split("_")[1]
        d = pd.read_csv(path)
        curves[sid] = [d[d.table == f"B{n}"].brier.mean() for n in BUDGETS]
    x = np.log2(np.array(BUDGETS) + 1)
    colours = [MUTED, GOOD, ACCENT, WARN]
    frames, durations = [], []
    for upto in range(1, 7):
        fig, ax = plt.subplots(figsize=(8.6, 4.4))
        for c, (sid, ys) in enumerate(curves.items()):
            ys = np.array(ys)
            ax.fill_between(x[:upto], ys[:upto], 0.0, color=colours[c % 4], alpha=0.10)
            ax.plot(x[:upto], ys[:upto], "o-", color=colours[c % 4], lw=2,
                    label=f"Codabench {sid}: ALC so far {np.dot(WEIGHTS[:upto], ys[:upto]) / sum(WEIGHTS[:upto]):.4f}")
        ax.axhline(0.25, color=WARN, ls="--", lw=1); ax.text(4.95, 0.252, "always 0.5", ha="right", fontsize=8, color=WARN)
        ax.set_xticks(x); ax.set_xticklabels([str(b) for b in BUDGETS])
        ax.set_xlim(-0.2, 5.2); ax.set_ylim(0.12, 0.27)
        ax.set_xlabel("Label budget n (log2(1+n) spacing)"); ax.set_ylabel("Mean Brier over sampled pairs")
        ax.set_title("Official formative feedback, built budget by budget"); ax.grid(color=GRID)
        ax.legend(fontsize=8, loc="lower left")
        fig.text(0.01, 0.01, "Measured on Codabench; each submission sees a new sample of 8-9 pairs, "
                 "so the curves are not a paired comparison.", fontsize=7.5, color=MUTED)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        frames.append(render(fig)); durations.append(2200 if upto == 6 else 900)
    save_gif(frames, out / "gif4_codabench_alc.gif", durations=durations)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=".")
    p.add_argument("--out", default="docs/figures")
    a = p.parse_args()
    repo, out = Path(a.repo), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_e1(repo, out)
    gif_item_peer(out)
    gif_codabench(repo, out)
