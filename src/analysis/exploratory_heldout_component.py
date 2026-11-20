"""Exploratory, not registered: does the first component separate drift at a fixed turn number?

    python src/analysis/exploratory_heldout_component.py --set main

PREREGISTRATION.md section 15, 2026-11-20. The design below was fixed and committed before this
script was run on the ratings, and after the results of exploratory_random_bar.py had been seen:
in emotional disclosure the first principal component of the activations separates judged drift
(AUC 0.82 in sample) and rises with the turn number (rank correlation 0.68). Two readings fit:
the component is position in the conversation, or it is drift building over the conversation.
This script asks one thing that bears on them: among turns at the SAME turn number, does the
component, fitted on other conversations, still separate drift turns from the rest?

Design, for one stratum at the target layer:

    fit          The first principal component of the activations of all turns of the OTHER
                 conversations of the stratum, mean removed. No rating enters the fit. Each
                 conversation is scored on the component fitted without it, centred on the mean
                 of the turns it was fitted on, and the projections are pooled across the fits.
    orientation  Each fitted component is given the sign that makes its cosine with the
                 released axis positive. The prediction is the axis's: drift turns have the
                 LOWER projection. The score is one-sided, AUC - 0.5, as for the axis.
    matching     Only pairs of a drift turn and a non-drift turn with the same turn number are
                 compared; the pair counts are added over turn numbers before dividing.
    interval     The conversation bootstrap of the registered tests (same number of resamples,
                 a stream of its own), with the fitted projections held fixed: the component is
                 not refitted inside a resample.

The headline is one number: that matched, held-out AUC in emotional disclosure, with its
interval. Everything else is there to read it against:

    two_by_two   the component in sample and held out, each unmatched and matched
    axis         the axis's AUC on the same pairs, and the component's AUC minus the axis's
    bars         the 95th percentile of the role directions' and of the random directions'
                 scores on the same pairs, and the component's and the axis's gap to each
    fits         the smallest cosine between two fitted components, and of each with the
                 component fitted on all conversations
    pairs        drift turns, non-drift turns and pairs at each turn number
    five_fold    sensitivity: five folds of conversations (seeded), the component fitted on four
                 and scored on the fifth, pairs only inside one fold at one turn number
    role_pc1     the absolute cosine of the component fitted on all conversations with the
                 first principal component of the released role vectors
    meta_reflection, as a second stratum: sensitivity only. Its first component went with
                 reply length, not turn number, in exploratory_random_bar.py.

Readings, fixed before the run. If the interval of the headline lies above 0.5: the component
separates judged drift among turns at the same position, so it is not only the turn number. That
is not "the component is drift": conversations at one turn number still differ in their brief
and in how far the user has pushed, and the judge is the criterion. If the interval includes
0.5: no evidence here that the component carries more than position. That is not "it is only
position": matching on the turn number also removes the drift that the briefs tie to the turn by
design, and the pairs are few. This test can speak against "only the turn number"; it cannot
confirm it.

Writes results/<set>/exploratory_heldout_component.json. Nothing here changes a registered figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyse_main import join_ratings, stream  # noqa: E402
from gap import PairCounts, auc_cells, pair_counts, resample_weights  # noqa: E402
from load_axis import load_config, load_released, unit  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
N_FOLDS = 5


def first_component(acts: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Mean, first principal component (unit length, positive cosine with `reference`) and its share of variance."""
    mean = acts.mean(axis=0)
    _, sv, vt = np.linalg.svd(acts - mean, full_matrices=False)
    v = vt[0] if vt[0] @ reference >= 0 else -vt[0]
    return mean, v, float(sv[0] ** 2 / (sv**2).sum())


def heldout_projections(acts: np.ndarray, group: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each turn projected on the first component fitted on the turns of every OTHER group.

    Returns the projections (n_turns,), the fitted components (n_groups, d) and their shares of variance."""
    proj = np.empty(len(acts))
    groups = np.unique(group)
    comps, shares = [], []
    for g in groups:
        out = group == g
        mean, v, share = first_component(acts[~out], reference)
        proj[out] = (acts[out] - mean) @ v
        comps.append(v)
        shares.append(share)
    return proj, np.array(comps), np.array(shares)


def cells_by(x: np.ndarray, drift: np.ndarray, conv: np.ndarray, n_conv: int, key: np.ndarray) -> list[PairCounts]:
    """Pair counts inside each value of `key` that has both a drift and a non-drift turn."""
    cells = []
    for k in np.unique(key):
        m = key == k
        if drift[m].any() and (~drift[m]).any():
            cells.append(pair_counts(x[m], drift[m], conv[m], n_conv))
    return cells


def interval(values: np.ndarray, lo: float, hi: float) -> list[float]:
    return [float(np.nanpercentile(values, lo)), float(np.nanpercentile(values, hi))]


def summarise(cells: list[PairCounts], w: np.ndarray, n_roles: int, cfg: dict) -> dict:
    """The columns of the cells are: component, axis, then the role directions, then the random directions."""
    pct, lo, hi = cfg["percentile"], cfg["lo"], cfg["hi"]
    auc, boot = auc_cells(cells), auc_cells(cells, w)
    out = {
        "pairs": float(sum(pc.n_drift.sum() * pc.n_non_drift.sum() for pc in cells)),
        "component_auc": float(auc[0]),
        "component_auc_ci": interval(boot[:, 0], lo, hi),
        "axis_auc": float(auc[1]),
        "axis_auc_ci": interval(boot[:, 1], lo, hi),
        "component_minus_axis": float(auc[0] - auc[1]),
        "component_minus_axis_ci": interval(boot[:, 0] - boot[:, 1], lo, hi),
    }
    for name, cols in (("role_directions", slice(2, 2 + n_roles)), ("random_directions", slice(2 + n_roles, None))):
        if auc[cols].size == 0:
            continue
        bar, bar_boot = np.percentile(np.abs(auc[cols] - 0.5), pct), np.percentile(np.abs(boot[:, cols] - 0.5), pct, axis=1)
        out[name] = {
            "bar": float(bar),
            "component_gap": float(auc[0] - 0.5 - bar),
            "component_gap_ci": interval(boot[:, 0] - 0.5 - bar_boot, lo, hi),
            "axis_gap": float(auc[1] - 0.5 - bar),
            "axis_gap_ci": interval(boot[:, 1] - 0.5 - bar_boot, lo, hi),
            "component_rank": int(1 + (np.abs(auc[cols] - 0.5) > auc[0] - 0.5).sum()),
        }
    return out


def analyse_stratum(
    acts: np.ndarray,
    others: np.ndarray,
    drift: np.ndarray,
    conv: np.ndarray,
    turn: np.ndarray,
    reference: np.ndarray,
    n_roles: int,
    w: np.ndarray,
    fold_of_conv: np.ndarray,
    cfg: dict,
) -> dict:
    """`others` holds the saved projections: the axis, the role directions, the random directions.
    `conv` runs from 0 to n_conv - 1; `fold_of_conv` gives each conversation's fold for the sensitivity."""
    n_conv = int(conv.max()) + 1
    mean, full, share = first_component(acts, reference)
    in_sample = (acts - mean) @ full
    held, comps, shares = heldout_projections(acts, conv, reference)
    fold = fold_of_conv[conv]
    held_fold, comps_fold, _ = heldout_projections(acts, fold, reference)
    one = np.zeros(len(turn), dtype=int)

    def table(component: np.ndarray, key: np.ndarray) -> dict:
        return summarise(cells_by(np.column_stack([component, others]), drift, conv, n_conv, key), w, n_roles, cfg)

    cos = comps @ comps.T
    return {
        "turns": int(len(turn)),
        "drift_turns": int(drift.sum()),
        "conversations": n_conv,
        "headline_heldout_matched": table(held, turn),
        "two_by_two": {
            "in_sample_unmatched": table(in_sample, one),
            "in_sample_matched": table(in_sample, turn),
            "heldout_unmatched": table(held, one),
            "heldout_matched": "see headline_heldout_matched",
        },
        "fits": {
            "variance_share_all_conversations": share,
            "variance_share_min_max": [float(shares.min()), float(shares.max())],
            "min_cosine_between_fits": float(cos.min()),
            "min_cosine_with_fit_on_all": float((comps @ full).min()),
            "cosine_with_axis_fit_on_all": float(full @ reference),
        },
        "pairs_by_turn_number": {
            int(t): {"drift": int(drift[turn == t].sum()), "non_drift": int((~drift[turn == t]).sum()), "pairs": int(drift[turn == t].sum() * (~drift[turn == t]).sum())}
            for t in np.unique(turn)
        },
        "five_fold": {
            "folds": N_FOLDS,
            "min_cosine_between_fits": float((comps_fold @ comps_fold.T).min()),
            "matched_within_fold": table(held_fold, fold * 1000 + turn),
        },
    }  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", required=True)
    args = parser.parse_args()
    cfg = load_config()
    out_dir = REPO_ROOT / "results" / args.set
    z = np.load(out_dir / "projections.npz")
    ratings_path = out_dir / f"ratings.rubric{cfg['rubric']['version']}.jsonl"
    rating, _, _ = join_ratings(z["conversation"], z["turn"], [json.loads(x) for x in ratings_path.read_text().splitlines() if x.strip()])
    layer = cfg["projection"]["target_layer"]
    li = z["layers"].tolist().index(layer)
    released = load_released(cfg)
    axis = unit(released.axis[layer])
    roles_centred = released.roles[:, layer] - released.roles[:, layer].mean(axis=0)
    role_pc1 = np.linalg.svd(roles_centred.astype(np.float64), full_matrices=False)[2][0]

    convs = list(dict.fromkeys(z["conversation"].tolist()))
    acts = np.concatenate([np.load(REPO_ROOT / "data" / "runs" / args.set / f"{c}.npy")[:, li] for c in convs]).astype(np.float64)
    use = z["has_activation"] & (rating >= 0)
    if not np.allclose((acts[use] @ axis), z["axis"][use, li], rtol=1e-4, atol=1e-3):
        raise SystemExit("the activations do not reproduce the saved axis projections")

    boot = {"percentile": cfg["off_target"]["percentile"], "lo": 2.5, "hi": 97.5}
    out: dict = {
        "exploratory": True,
        "registered_in": "PREREGISTRATION.md section 15, 2026-11-20",
        "layer": layer,
        "headline": "strata.emotional_disclosure.headline_heldout_matched.component_auc",
        "bootstrap": {"n_resamples": cfg["bootstrap"]["n_resamples"], "refit_inside_resample": False},
        "strata": {},
    }
    for s in ("emotional_disclosure", "meta_reflection"):
        m = use & (z["stratum"] == s)
        _, conv = np.unique(z["conversation"][m], return_inverse=True)
        n_conv = int(conv.max()) + 1
        rng = stream(cfg["seeds"]["bootstrap"], f"heldout_component:{s}")
        w = resample_weights(n_conv, cfg["bootstrap"]["n_resamples"], rng)
        fold_of_conv = stream(cfg["seeds"]["bootstrap"], f"heldout_component_folds:{s}").permutation(n_conv) % N_FOLDS
        others = np.column_stack([z["axis"][m, li], z["roles"][m, li, :], z["random"][m, li, :]]).astype(np.float64)
        res = analyse_stratum(acts[m], others, rating[m] >= 2, conv, z["turn"][m], axis, z["roles"].shape[2], w, fold_of_conv, boot)
        _, full, _ = first_component(acts[m], axis)
        res["role_pc1_abs_cosine_fit_on_all"] = float(abs(full @ role_pc1))
        res["role_pc1_abs_cosine_with_axis"] = float(abs(axis @ role_pc1))
        res["standing"] = "headline" if s == "emotional_disclosure" else "sensitivity"
        out["strata"][s] = res
    (out_dir / "exploratory_heldout_component.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
