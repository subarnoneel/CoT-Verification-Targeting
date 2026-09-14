#!/usr/bin/env python3
"""
Regenerate every figure in the report and presentation from the analysis tables.

USAGE
    python src/make_figures.py

Reads   data/results/analysis_table_FINAL.csv
        data/results/e4_scored.csv        (optional; skips the E4 figure if absent)
Writes  figures/fig_initiation_vs_hit_FINAL.png
        figures/fig_accuracy_FINAL.png
        figures/fig_e4_agreement.png

No GPU and no network needed. Runs in a few seconds.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "data", "results")
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ASTRA presentation palette — keep these in sync with beamerthemeASTRA.sty
NAVY, BLUE, ORANGE, GREY = "#0D2B5E", "#156082", "#E97132", "#9BB3C9"
LABELS = {"C1": "Arithmetic\nerror", "C2": "Authority\nclaim", "C4": "Contradiction"}


def load_analysis():
    path = os.path.join(RESULTS, "analysis_table_FINAL.csv")
    df = pd.read_csv(path)
    return df[df.continuation_mode == "continued"]


def fig_initiation_vs_hit(d):
    """Keyword baseline against the graph measure, per damage type."""
    sub = d[d.condition.isin(["C1", "C2", "C4"])].dropna(subset=["audit_hit"])
    g = sub.groupby("condition").agg(cue=("cue_present", "mean"),
                                     hit=("audit_hit", "mean"))
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    x, w = np.arange(len(g)), 0.36
    ax.bar(x - w / 2, g.cue, w, label="Says it is checking (keyword method)", color=BLUE)
    ax.bar(x + w / 2, g.hit, w, label="Actually checks the damaged step (graph)", color=ORANGE)
    for i, (c, h) in enumerate(zip(g.cue, g.hit)):
        ax.text(i - w / 2, c + 0.02, f"{c:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, h + 0.02, f"{h:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[i] for i in g.index])
    ax.set_ylabel("Proportion of chains")
    ax.set_ylim(0, 1.12)
    ax.set_title("Checking happens far more often than correct checking",
                 color=NAVY, fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIGDIR, "fig_initiation_vs_hit_FINAL.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print("wrote", os.path.relpath(out, ROOT))


def fig_funnel(d):
    """Initiation / targeting / repair side by side — the funnel dissociation."""
    sub = d[d.condition.isin(["C1", "C2", "C4"])].dropna(subset=["audit_hit"])
    f = sub.groupby("condition").agg(b1=("check_initiation", "mean"),
                                     b2=("audit_hit", "mean"),
                                     b4=("final_correct", "mean"))
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    x, w = np.arange(len(f)), 0.26
    ax.bar(x - w, f.b1, w, label="B1 Initiation (checks at all)", color=GREY)
    ax.bar(x, f.b2, w, label="B2 Targeting (checks the damage)", color=BLUE)
    ax.bar(x + w, f.b4, w, label="B4 Repair (final answer correct)", color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[i] for i in f.index])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Proportion")
    ax.set_title("The verification funnel separates abilities accuracy cannot",
                 color=NAVY, fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIGDIR, "fig_accuracy_FINAL.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print("wrote", os.path.relpath(out, ROOT))


def fig_e4():
    """Labelling agreement on clean vs damaged text (H4)."""
    path = os.path.join(RESULTS, "e4_scored.csv")
    if not os.path.exists(path):
        print("skipped fig_e4_agreement.png (e4_scored.csv not found)")
        return
    from sklearn.metrics import cohen_kappa_score
    m = pd.read_csv(path)
    m["human"] = np.where(m.A == m.B, m.A, m.A)
    clean, dmg = m[~m.is_damaged], m[m.is_damaged]

    vals = [
        cohen_kappa_score(m.A, m.B),
        cohen_kappa_score(clean.model, clean.human),
        cohen_kappa_score(dmg.model, dmg.human),
    ]
    labels = ["Human\nvs human", "Model vs human\nCLEAN text", "Model vs human\nDAMAGED text"]
    cols = [GREY, BLUE, ORANGE]

    # the same comparison with our own injected steps removed
    if "is_injected" in dmg.columns:
        di = dmg[~dmg.is_injected]
        vals.append(cohen_kappa_score(di.model, di.human))
        labels.append("Model vs human\nDAMAGED,\nexcl. injected steps")
        cols.append(BLUE)

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.bar(range(len(vals)), vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0.60, ls="--", lw=1.2, color="#B00020")
    ax.text(len(vals) - 0.48, 0.615, "acceptance\nthreshold 0.60",
            fontsize=7.5, color="#B00020", ha="right")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Cohen's kappa")
    ax.set_ylim(0, 1.10)
    ax.set_title("Labelling agreement holds on damaged text", color=NAVY, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIGDIR, "fig_e4_agreement.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    d = load_analysis()
    print(f"loaded {len(d)} usable chains")
    fig_initiation_vs_hit(d)
    fig_funnel(d)
    fig_e4()
    print("done")
