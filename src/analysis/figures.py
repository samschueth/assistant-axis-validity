"""Every figure in WRITEUP.md, from files in results/.

    python src/analysis/figures.py --set main

Writes results/<set>/figures/*.png. Figures 2 and 3 are drawn from analysis.json and
descriptions.json alone. Figure 1 needs each direction's score, which those files hold only as
quantiles, so it recomputes the scores from projections.npz and the ratings with the same
functions the analysis uses; the bar and the axis's score it draws are checked against
analysis.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analyse_main as am  # noqa: E402
from gap import auc_lower, gap_from_auc, pair_counts  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"
NAMES = {"task": "Task", "meta_reflection": "Meta-reflection", "emotional_disclosure": "Emotional disclosure"}
COLOURS = {"task": AQUA, "meta_reflection": ORANGE, "emotional_disclosure": BLUE}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    }
)  # fmt: skip


def figure1(set_name: str, analysis: dict, out: Path) -> None:
    """The axis's score against the scores of the off-target directions, in each stratum that enters."""
    with open(REPO_ROOT / "configs" / "study.toml", "rb") as f:
        study = tomllib.load(f)
    res = REPO_ROOT / "results" / set_name
    z = np.load(res / "projections.npz")
    ratings = [json.loads(x) for x in (res / analysis["inputs"]["ratings"]).read_text().splitlines() if x.strip()]
    rating, _, _ = am.join_ratings(z["conversation"], z["turn"], ratings)
    li = z["layers"].tolist().index(analysis["inputs"]["target_layer"])
    use = z["has_activation"] & (rating >= 0)
    strata = [s for s, r in analysis["primary"]["role_directions"]["strata"].items() if r["enters"]]
    strata.sort(key=study["sample"]["strata"].index, reverse=True)
    bars = (("roles", "role_directions", "275 role directions"), ("random", "random_directions", "1,000 random directions"))
    fig, axes = plt.subplots(len(bars), len(strata), figsize=(7.2, 4.6), sharex=True, squeeze=False)
    for row, (key, name, label) in enumerate(bars):
        for col, s in enumerate(strata):
            m = use & (z["stratum"] == s)
            x = np.column_stack([z["axis"][m, li], z[key][m, li, :]]).astype(np.float64)
            g = gap_from_auc(auc_lower(pair_counts(x, rating[m] >= 2, z["conversation"][m])), study["off_target"]["percentile"])
            want = analysis["primary"][name]["strata"][s]
            assert np.isclose(g["bar"], want["bar"]) and np.isclose(g["target_score"] + 0.5, want["target_auc"])
            ax = axes[row, col]
            ax.hist(g["off_scores"], bins=np.arange(0, 0.36, 0.0125), color=NEUTRAL, edgecolor=SURFACE, linewidth=0.6)
            top = ax.get_ylim()[1]
            ax.axvline(g["bar"], color=MUTED, linewidth=1.2, linestyle=(0, (4, 2)))
            ax.axvline(g["target_score"], color=BLUE, linewidth=2)
            ax.text(g["bar"] - 0.004, top * 1.02, f"95th pct. {g['bar']:.2f}", color=MUTED, ha="right", va="bottom", fontsize=8)
            ax.text(g["target_score"] + 0.004, top * 1.16, f"axis {g['target_score']:.2f}", color=INK, ha="left", va="bottom", fontsize=8)
            ax.set_ylim(0, top * 1.32)
            ax.set_title(f"{NAMES[s]}: {label}" if row == 0 else label, fontsize=9)
            ax.set_ylabel("directions")
            if row == len(bars) - 1:
                ax.set_xlabel("score: distance of the AUC from 0.5")
    fig.tight_layout()
    fig.savefig(out / "fig1_off_target_distributions.png", dpi=200)
    plt.close(fig)


def figure2(analysis: dict, out: Path) -> None:
    """The gap with its interval: the registered test, its parts, the pooled figure and the checks."""
    p, layers = analysis["primary"], analysis["layers"]
    target = analysis["inputs"]["target_layer"]
    roles = p["role_directions"]
    rows = [("Within type, averaged (the registered test)", roles["primary"], True)]
    rows += [(f"  {NAMES[s]} alone", r, False) for s, r in roles["strata"].items() if r["enters"]]
    rows += [("Pooled over all turns", roles["pooled"], False)]
    rows += [("Within type, against random directions", p["random_directions"]["primary"], False)]
    rows += [(f"Within type, layer {k} (target is {target})", layers[k]["primary"], False) for k in sorted(layers, key=int)]
    fig, ax = plt.subplots(figsize=(7.6, 0.36 * len(rows) + 1.0))
    for i, (label, r, main) in enumerate(rows):
        y = len(rows) - 1 - i
        colour = BLUE if main else MUTED
        ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=colour, linewidth=2.4 if main else 1.6, solid_capstyle="round")
        ax.plot(r["gap"], y, "o", color=colour, markersize=7 if main else 5.5, markeredgecolor=SURFACE, markeredgewidth=1)
        ax.text(0.17, y, f"{r['gap']:+.3f}  [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]", va="center", ha="left", fontsize=8,
                color=INK if main else MUTED, family="DejaVu Sans Mono")  # fmt: skip
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], color=INK)
    ax.set_xlim(-0.16, 0.40)
    ax.set_xticks([-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15])
    ax.set_xlabel("gap, with its 95% interval", loc="left")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "fig2_gaps.png", dpi=200)
    plt.close(fig)


def figure3(analysis: dict, descriptions: dict, out: Path) -> None:
    """The projection and the share of drift turns, by turn number: both move with position."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    for s in descriptions["strata"]:
        mean = descriptions["strata"][s]["item3_course_over_turns"]["mean_projection_by_turn_number"]
        turns = sorted(mean, key=int)
        top.plot([int(k) for k in turns], [mean[k] for k in turns], color=COLOURS[s], linewidth=2, marker="o", markersize=3.5)
        top.text(15.25, mean[turns[-1]], NAMES[s], color=INK, va="center", fontsize=8.5)
        by = analysis["drift_counts"][s]["drift_by_turn_number"]
        share = [100 * by[k][0] / by[k][1] for k in turns]
        bottom.plot([int(k) for k in turns], share, color=COLOURS[s], linewidth=2, marker="o", markersize=3.5, label=NAMES[s])
        bottom.text(15.25, share[-1], NAMES[s], color=INK, va="center", fontsize=8.5)
    top.set_title("Mean projection on the Assistant Axis")
    top.set_ylabel("projection (layer 32)")
    bottom.set_title("Share of turns the judge rated as drift")
    bottom.set_ylabel("% of turns")
    bottom.set_xlabel("assistant turn in the conversation")
    bottom.set_xticks(range(1, 16))
    bottom.set_xlim(0.5, 19.2)
    bottom.set_ylim(-3, 100)
    for ax in (top, bottom):
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for edge in (5.5, 10.5):  # every brief changes course after turns 5 and 10
            ax.axvline(edge, color=NEUTRAL, linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
    bottom.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_course_over_turns.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", required=True)
    args = parser.parse_args()
    res = REPO_ROOT / "results" / args.set
    out = res / "figures"
    out.mkdir(exist_ok=True)
    analysis = json.loads((res / "analysis.json").read_text())
    descriptions = json.loads((res / "descriptions.json").read_text())
    figure1(args.set, analysis, out)
    figure2(analysis, out)
    figure3(analysis, descriptions, out)
    print("\n".join(str(p.relative_to(REPO_ROOT)) for p in sorted(out.glob("*.png"))))


if __name__ == "__main__":
    main()
