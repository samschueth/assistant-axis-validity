"""Sample-size simulation: can about 1,350 turns detect a gap of plausible size?

Everything here is synthetic. No value in the scenario grid comes from any data: the grid spans
assumptions, and the output says what the design can and cannot detect under each. The analysis
applied to each simulated dataset is the real one in gap.py, with the pre-registered stratum
entry rule and decision rule.

Generative model, per replicate
  - 3 strata x 30 conversations x 15 assistant turns.
  - Drift: a latent score with a conversation-level component (intraclass correlation icc_y) is
    cut at a threshold that gives the stratum's base rate. Drift therefore clusters in
    conversations.
  - Projections: for each direction, drift turns are shifted down by the amount that gives its
    true AUC; the noise has a conversation-level component (icc_x). Off-target noise is
    correlated across directions through a few shared factors, because persona directions are
    not independent of each other.
  - Off-target directions: 275, each with a true signed AUC deviation drawn from a normal
    distribution scaled so that the 95th percentile of its absolute value is `off_p95`.
  - Target: a fixed true AUC, or ("exchangeable") a true deviation drawn from the off-target
    distribution with the favourable sign, which is the worst case for a false positive.

    python src/analysis/sample_size_sim.py            # full grid
    python src/analysis/sample_size_sim.py --smoke    # two replicates per scenario, no output file
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import tomllib
from multiprocessing import Pool
from pathlib import Path
from statistics import NormalDist

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gap import analyse  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_CSV = REPO_ROOT / "results" / "sample_size_sim.csv"
OUT_MD = REPO_ROOT / "results" / "sample_size_sim.md"

N_REPLICATES = 1000
N_RESAMPLES = 1000  # per replicate; the real analysis uses 2,000
N_FACTORS = 10
FACTOR_SHARE = 0.5  # share of off-target noise variance carried by the shared factors
STRATA = ("task", "meta_reflection", "emotional_disclosure")

BASE_RATES = {
    "higher": (0.05, 0.35, 0.35),
    "lower": (0.02, 0.15, 0.15),
}
TARGET_AUCS = (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
OFF_P95 = (0.00, 0.05, 0.10)
CLUSTERING = {
    "moderate": (0.4, 0.3),  # (icc_y, icc_x)
    "strong": (0.6, 0.5),
}


def scenario_grid() -> list[dict]:
    grid = []
    for rates, off, auc in itertools.product(BASE_RATES, OFF_P95, TARGET_AUCS):
        grid.append({"base_rates": rates, "off_p95": off, "target": auc, "clustering": "moderate"})
    for rates, off in itertools.product(BASE_RATES, (0.05, 0.10)):
        grid.append({"base_rates": rates, "off_p95": off, "target": "exchangeable", "clustering": "moderate"})
    for auc in TARGET_AUCS:  # sensitivity to clustering, at the central setting
        grid.append({"base_rates": "higher", "off_p95": 0.05, "target": auc, "clustering": "strong"})
    return grid


_N = NormalDist()


def shift_for_auc(auc: np.ndarray) -> np.ndarray:
    """Mean shift, in within-class SD units, that gives this AUC for two unit normals."""
    return np.sqrt(2.0) * np.vectorize(_N.inv_cdf)(auc)


def clustered_noise(rng, n_conv, n_turns, n_dir, icc, shared: bool) -> np.ndarray:
    """Unit-variance noise of shape (n_conv * n_turns, n_dir), clustered by conversation."""

    def field(rows: int) -> np.ndarray:
        e = rng.standard_normal((rows, n_dir))
        if not shared:
            return e
        loadings = rng.standard_normal((N_FACTORS, n_dir))
        loadings /= np.linalg.norm(loadings, axis=0, keepdims=True)
        f = rng.standard_normal((rows, N_FACTORS)) @ loadings
        return np.sqrt(FACTOR_SHARE) * f + np.sqrt(1 - FACTOR_SHARE) * e

    conv_part = np.repeat(field(n_conv), n_turns, axis=0)
    return np.sqrt(icc) * conv_part + np.sqrt(1 - icc) * field(n_conv * n_turns)


def simulate_once(sc: dict, cfg: dict, rng: np.random.Generator) -> dict:
    n_conv = cfg["sample"]["conversations_per_stratum"]
    n_turns = cfg["sample"]["assistant_turns"]
    n_dir = cfg["vectors"]["n_role_vectors"]
    icc_y, icc_x = CLUSTERING[sc["clustering"]]

    tau = sc["off_p95"] / _N.inv_cdf(0.975)
    off_auc = np.clip(0.5 + tau * rng.standard_normal(n_dir), 0.02, 0.98)
    if sc["target"] == "exchangeable":
        target_auc = float(np.clip(0.5 + abs(tau * rng.standard_normal()), 0.02, 0.98))
    else:
        target_auc = float(sc["target"])
    shifts = shift_for_auc(np.concatenate([[target_auc], off_auc]))

    xs, ys, convs, strata = [], [], [], []
    for s, (name, rate) in enumerate(zip(STRATA, BASE_RATES[sc["base_rates"]])):
        latent = np.sqrt(icc_y) * np.repeat(rng.standard_normal(n_conv), n_turns) + np.sqrt(
            1 - icc_y
        ) * rng.standard_normal(n_conv * n_turns)
        drift = latent > _N.inv_cdf(1 - rate)
        noise = np.column_stack(
            [
                clustered_noise(rng, n_conv, n_turns, 1, icc_x, shared=False),
                clustered_noise(rng, n_conv, n_turns, n_dir, icc_x, shared=True),
            ]
        )
        xs.append(noise - drift[:, None] * shifts[None, :])
        ys.append(drift)
        convs.append(s * n_conv + np.repeat(np.arange(n_conv), n_turns))
        strata.append(np.full(n_conv * n_turns, name))
    x = np.vstack(xs)

    res = analyse(
        x[:, 0],
        x[:, 1:],
        np.concatenate(ys),
        np.concatenate(convs),
        np.concatenate(strata),
        min_drift=cfg["sample"]["min_drift_turns"],
        min_non_drift=cfg["sample"]["min_non_drift_turns"],
        percentile=cfg["off_target"]["percentile"],
        n_resamples=N_RESAMPLES,
        ci_level=cfg["bootstrap"]["ci_level"],
        rng=rng,
        pooled=False,
    )
    entering = [res["strata"][s] for s in STRATA if res["strata"][s]["enters"]]
    p = res["primary"]
    return {
        "n_entering": res["n_strata_entering"],
        **{f"enters_{s}": res["strata"][s]["enters"] for s in STRATA},
        "estimable": p is not None,
        "bar": np.mean([r["bar"] for r in entering]) if entering else np.nan,
        "target_auc_hat": np.mean([r["target_auc"] for r in entering]) if entering else np.nan,
        "gap": p["gap"] if p else np.nan,
        "above_zero": bool(p and p["above_zero"]),
        "point_above_zero": bool(p and p["gap"] > 0),
    }


def run_scenario(args: tuple[int, dict, dict, int, int]) -> dict:
    index, sc, cfg, seed, n_rep = args
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    reps = [simulate_once(sc, cfg, rng) for _ in range(n_rep)]

    def mean(key: str) -> float:
        return float(np.nanmean([r[key] for r in reps]))

    return {
        **sc,
        "n_replicates": n_rep,
        "p_estimable": mean("estimable"),
        **{f"p_enters_{s}": mean(f"enters_{s}") for s in STRATA},
        "mean_bar": mean("bar"),
        "mean_target_auc_hat": mean("target_auc_hat"),
        "mean_gap": mean("gap"),
        "p_point_gap_above_zero": mean("point_above_zero"),
        # Not-estimable replicates count as "not above zero": this is power for the whole design.
        "p_gap_above_zero": mean("above_zero"),
    }


def write_summary(rows: list[dict]) -> None:
    def fmt(v) -> str:
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    cols = [
        ("base_rates", "base rates"),
        ("off_p95", "true off-target P95"),
        ("clustering", "clustering"),
        ("target", "true target AUC"),
        ("p_estimable", "P(estimable)"),
        ("mean_bar", "mean bar"),
        ("mean_gap", "mean gap"),
        ("p_gap_above_zero", "P(gap above zero)"),
    ]
    lines = [
        "# Sample-size simulation",
        "",
        "Generated by `src/analysis/sample_size_sim.py`. Synthetic data only; see the script for",
        "the generative model. Base rates are drift rates in the task, meta-reflection and",
        "emotional-disclosure strata: "
        + "; ".join(f"{k} = {v}" for k, v in BASE_RATES.items())
        + ".",
        f"{N_REPLICATES} replicates per scenario, {N_RESAMPLES} bootstrap resamples per replicate.",
        "",
        "- **bar**: the 95th percentile of the 275 off-target scores, averaged over entering strata.",
        "  With a true off-target P95 of 0.00 it is pure sampling noise: the floor the target must clear.",
        "- **P(gap above zero)**: share of replicates in which the lower bound of the 95% bootstrap",
        "  interval for the primary gap is above zero. With a true target AUC of 0.50 or an",
        "  exchangeable target this is the false-positive rate; otherwise it is power.",
        "",
        "| " + " | ".join(h for _, h in cols) + " |",
        "|" + "---|" * len(cols),
    ]
    lines += ["| " + " | ".join(fmt(r[k]) for k, _ in cols) + " |" for r in rows]
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="two replicates per scenario, no files")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    with open(REPO_ROOT / "configs" / "study.toml", "rb") as f:
        cfg = tomllib.load(f)
    n_rep = 2 if args.smoke else N_REPLICATES
    jobs = [(i, sc, cfg, cfg["seeds"]["simulation"], n_rep) for i, sc in enumerate(scenario_grid())]
    with Pool(args.workers) as pool:
        rows = pool.map(run_scenario, jobs, chunksize=1)

    if args.smoke:
        print(f"smoke run completed: {len(rows)} scenarios x {n_rep} replicates")
        return
    OUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    write_summary(rows)
    print(f"wrote {OUT_CSV.relative_to(REPO_ROOT)} and {OUT_MD.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
