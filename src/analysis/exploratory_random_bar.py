"""Exploratory, not registered: why the random directions score as high as they do.

    python src/analysis/exploratory_random_bar.py --set main

Written on 2026-11-19 after the registered analyses had been run and seen. In emotional
disclosure the 95th percentile of the random directions' scores (0.249) is above that of the role
directions (0.190), although random directions were meant as the easier bar. This script asks
whether the activations in a stratum vary mostly along a few components that also separate drift,
so that every random direction picks up a share of them.

For each stratum that enters the primary test, at the target layer, from the saved activations
(data/runs/<set>/, not committed) with the stratum's mean subtracted:

    spectrum        the share of variance along each of the first ten principal components, and
                    the participation ratio (the effective number of components)
    components      for each of the first ten components: its cosine with the axis, and the
                    distance from 0.5 of its AUC for drift
    random          the 95th percentile of the 1,000 random directions' scores as registered,
                    then with the axis component removed from each direction, and then with the
                    first k principal components removed (k = 1, 3, 10)
    roles           the same for the 275 role directions
    axis            the axis's score as registered, and with the first k components removed
    first_component what the first component goes with: its rank correlation with the turn number,
                    with the length of the reply in characters and with the axis projection

Writes results/<set>/exploratory_random_bar.json. Nothing here changes a registered figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyse_main import join_ratings  # noqa: E402
from load_axis import load_config, load_released, random_directions, role_directions, unit  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def auc_lower(x: np.ndarray, drift: np.ndarray) -> np.ndarray:
    """For each column of x: the probability that a drift turn is lower than a non-drift turn."""
    out = []
    for j in range(x.shape[1]):
        v = x[:, j]
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(1, len(v) + 1)
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r)
        r = (sums / counts)[inv]
        n1, n0 = drift.sum(), (~drift).sum()
        u = r[drift].sum() - n1 * (n1 + 1) / 2  # pairs in which the drift turn is HIGHER, ties one half
        out.append(1 - u / (n1 * n0))
    return np.array(out)


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def remove(directions: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Each direction with its components along the orthonormal rows of `basis` removed, at unit length."""
    d = directions - (directions @ basis.T) @ basis
    return d / np.linalg.norm(d, axis=1, keepdims=True)


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
    roles = role_directions(released.roles[:, layer], released.axis[layer])
    rand = random_directions(cfg["off_target"]["n_random_directions"], axis.shape[-1], cfg["seeds"]["random_directions"])
    pct = cfg["off_target"]["percentile"]

    convs = list(dict.fromkeys(z["conversation"].tolist()))
    acts = np.concatenate([np.load(REPO_ROOT / "data" / "runs" / args.set / f"{c}.npy")[:, li] for c in convs]).astype(np.float64)
    use = z["has_activation"] & (rating >= 0)
    reply = {
        (c["id"], k): len(t["reply"])
        for c in map(json.loads, (out_dir / "conversations.jsonl").read_text().splitlines())
        for k, t in enumerate(c["turns"], start=1)
    }
    length = np.array([reply[key] for key in zip(z["conversation"].tolist(), z["turn"].tolist())])
    if not np.allclose((acts[use] @ axis), z["axis"][use, li], rtol=1e-4, atol=1e-3):
        raise SystemExit("the activations do not reproduce the saved axis projections")

    out: dict = {"layer": layer, "strata": {}}
    for s in cfg["sample"]["strata"]:
        m = use & (z["stratum"] == s)
        drift = rating[m] >= 2
        if drift.sum() < 15 or (~drift).sum() < 15:
            continue
        a = acts[m] - acts[m].mean(axis=0)
        _, sv, vt = np.linalg.svd(a, full_matrices=False)
        var = sv**2 / (sv**2).sum()
        pcs = vt[:10]

        def bar(directions: np.ndarray) -> float:
            return float(np.percentile(np.abs(auc_lower(a @ directions.T, drift) - 0.5), pct))

        def score(direction: np.ndarray) -> float:
            return float(auc_lower((a @ direction)[:, None], drift)[0] - 0.5)

        out["strata"][s] = {
            "turns": int(m.sum()),
            "drift_turns": int(drift.sum()),
            "spectrum": {
                "variance_share_first_ten": [round(float(v), 4) for v in var[:10]],
                "participation_ratio": float(1 / (var**2).sum()),
            },
            "components": [
                {
                    "component": k + 1,
                    "cosine_with_axis": float(pcs[k] @ axis),
                    "auc_distance_from_half": float(abs(score(pcs[k]))),
                }
                for k in range(10)
            ],
            "first_component": {
                "rank_correlation_with_turn_number": rank_corr(a @ pcs[0], z["turn"][m]),
                "rank_correlation_with_reply_length": rank_corr(a @ pcs[0], length[m]),
                "rank_correlation_with_axis_projection": rank_corr(a @ pcs[0], a @ axis),
                "axis_rank_correlation_with_turn_number": rank_corr(a @ axis, z["turn"][m]),
                "axis_rank_correlation_with_reply_length": rank_corr(a @ axis, length[m]),
            },
            "random": {
                "bar_as_registered": bar(rand),
                "bar_axis_removed": bar(remove(rand, axis[None, :])),
                **{f"bar_first_{k}_components_removed": bar(remove(rand, pcs[:k])) for k in (1, 3, 10)},
                "mean_abs_cosine_with_axis": float(np.abs(rand @ axis).mean()),
            },
            "roles": {
                "bar_as_registered": bar(roles),
                **{f"bar_first_{k}_components_removed": bar(remove(roles, pcs[:k])) for k in (1, 3, 10)},
            },
            "axis": {
                "score_as_registered": score(axis),
                **{f"score_first_{k}_components_removed": score(remove(axis[None, :], pcs[:k])[0]) for k in (1, 3, 10)},
            },
        }
    (out_dir / "exploratory_random_bar.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
