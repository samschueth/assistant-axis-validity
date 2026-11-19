"""The registered analyses of the main run (PREREGISTRATION.md sections 4, 10, 15 and 16.2).

    python src/analysis/analyse_main.py --set main

Reads results/<set>/projections.npz, results/<set>/ratings.rubric<N>.jsonl (N from
configs/study.toml), labels/sample_<set>.json and the human labels. Writes
results/<set>/analysis.json. Calls no API.

What is computed, in the order of section 10:

    drift_counts      item 3: ratings and drift turns by stratum and by turn number
    primary           items 4 and 5: target AUC, off-target bar, rank and gap at the target layer,
                      within stratum and pooled, against the role directions and against the
                      random directions
    layers            item 6: the same against the role directions at the reliability layers
    lagged            section 16.2: whether the projection on a normal turn anticipates drift on
                      the next
    human_labels      section 15 (i): the projection against the owner's labels, a description
    strict_cut        section 15 (j): the primary test with a drift turn cut at a rating of 3
    sign_split        section 15 (l): the judge's drift turns split by sign

The descriptions of section 16.3 are not computed here.

Choices the registered text leaves open, fixed here before this script was run on the ratings:

- Ratings and projections are joined on (conversation, turn), never on row order.
- A turn with no activation is left out of everything that uses its projection. It can still
  be the second turn of a lagged pair (section 16.1).
- Every analysis has its own stream of random numbers, derived from seeds.bootstrap and the
  analysis's name, so that adding an analysis cannot change another's interval.
- 16.2, rival predictor: the judge's rating of turn k is scored in its natural direction. Its
  score is the probability that a drift-next pair has the HIGHER rating at turn k than a
  normal-next pair it is compared with, ties as one half, minus 0.5.
- 16.2, "pairs with the same rating at turn k": the cells are band by rating at turn k.
- 15 (i): on all 120 labelled turns, the owner's labels are those of version 3 for items 41 to
  120 and of version 2 for the pilot's items 1 to 40; the judge's ratings set beside them are
  those of the final version for every turn.
- 15 (l): the AUC between the two kinds of drift turn is the probability that a drift turn with
  sign A has a LOWER projection than a drift turn with C alone, ties as one half. The interval
  for a single stratum, and the difference between the two gaps with its interval, are not in
  the registered text and are labelled exploratory. Both splits in a stratum use the same
  bootstrap draws.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gap import analyse, auc_cells, gap_from_auc, pair_counts, resample_weights  # noqa: E402
from label import labels_name, latest_labels  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BANDS = ((1, 5), (6, 10), (11, 15))  # section 16.1


def stream(seed: int, name: str) -> np.random.Generator:
    """A random stream of its own for the analysis called `name`."""
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


def join_ratings(conversation: np.ndarray, turn: np.ndarray, rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ratings and signs in the order of the projections, matched on (conversation, turn).

    Returns rating (int, -1 where the judge gave none), sign_a and sign_c (bool)."""
    by_key = {(r["id"], int(r["turn"])): r for r in rows}
    if len(by_key) != len(rows):
        raise SystemExit("the ratings file rates a turn more than once")
    keys = list(zip(conversation.tolist(), turn.tolist()))
    if set(keys) != set(by_key):
        raise SystemExit("the rated turns are not the turns of the projections file")
    got = [by_key[k] for k in keys]
    rating = np.array([-1 if r["rating"] is None else r["rating"] for r in got], dtype=int)
    sign_a = np.array(["A" in r.get("signs", []) for r in got], dtype=bool)
    sign_c = np.array(["C" in r.get("signs", []) for r in got], dtype=bool)
    return rating, sign_a, sign_c


def drift_counts(stratum: np.ndarray, turn: np.ndarray, rating: np.ndarray, strata: list[str]) -> dict:
    out = {}
    for s in strata:
        m = (stratum == s) & (rating >= 0)
        out[s] = {
            "turns": int(m.sum()),
            "ratings": {str(r): int((rating[m] == r).sum()) for r in range(4)},
            "drift_turns": int((rating[m] >= 2).sum()),
            "drift_by_turn_number": {
                str(k): [int(((rating >= 2) & m & (turn == k)).sum()), int((m & (turn == k)).sum())]
                for k in sorted(set(turn[m].tolist()))
            },
        }
    return out


def cell_stats(cells: list, weights: np.ndarray, percentile: float, lo: float, hi: float) -> dict:
    """Target AUC, bar, gap and rank for pair counts in one stratum, with the bootstrap gaps.

    Column 0 is the target and the rest are off-target directions. `boot` is for combining
    strata and is removed before anything is written."""
    g = gap_from_auc(auc_cells(cells), percentile)
    boot = gap_from_auc(auc_cells(cells, weights), percentile)["gap"]
    ci = np.nanpercentile(boot, [lo, hi])
    return {
        "target_auc": float(g["target_score"] + 0.5),
        "bar": float(g["bar"]),
        "gap": float(g["gap"]),
        "rank": int(1 + (g["off_scores"] >= g["target_score"]).sum()),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "boot": boot,
    }


def average(rows: dict[str, dict], lo: float, hi: float) -> dict | None:
    """The equal-weight mean of the gap over strata, with its interval. None under two strata.
    A resample in which a stratum has nothing to compare drops that stratum for that draw."""
    if len(rows) < 2:
        return None
    boot = np.nanmean(np.vstack([r["boot"] for r in rows.values()]), axis=0)
    ci = np.nanpercentile(boot, [lo, hi])
    return {
        "strata": sorted(rows),
        "gap": float(np.mean([r["gap"] for r in rows.values()])),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "above_zero": bool(ci[0] > 0),
    }


def public(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "boot"}


def lagged_pairs(conversation: np.ndarray, turn: np.ndarray, rating: np.ndarray, has_activation: np.ndarray):
    """Rows (k, k+1) of the lagged pairs of section 16.1: turn k rated normal with an activation,
    turn k+1 of the same conversation rated."""
    row_of = {key: i for i, key in enumerate(zip(conversation.tolist(), turn.tolist()))}
    first, second = [], []
    for i, (c, t) in enumerate(zip(conversation.tolist(), turn.tolist())):
        j = row_of.get((c, t + 1))
        if j is not None and rating[i] in (0, 1) and has_activation[i] and rating[j] >= 0:
            first.append(i)
            second.append(j)
    return np.array(first, dtype=int), np.array(second, dtype=int)


def band_of(turn: np.ndarray) -> np.ndarray:
    band = np.full(len(turn), -1)
    for b, (lo, hi) in enumerate(BANDS):
        band[(turn >= lo) & (turn <= hi)] = b
    return band


def lagged(d: dict, x: np.ndarray, cfg: dict, rng: np.random.Generator) -> dict:
    """Section 16.2. `x` is (n_turns, 1 + n_off_target) at the target layer."""
    lo, hi, pct, n_res = cfg["lo"], cfg["hi"], cfg["percentile"], cfg["n_resamples"]
    first, second = lagged_pairs(d["conversation"], d["turn"], d["rating"], d["has_activation"])
    drift_next = d["rating"][second] >= 2
    band = band_of(d["turn"][first])
    rating_k = d["rating"][first]
    out: dict = {"strata": {}}
    banded, unbanded, same_rating = {}, {}, {}
    for s in cfg["strata"]:
        m = d["stratum"][first] == s
        convs, conv = np.unique(d["conversation"][first][m], return_inverse=True)
        n1, n0 = int(drift_next[m].sum()), int((~drift_next[m]).sum())
        n_conv_drift = len(set(conv[drift_next[m]].tolist()))
        enters = n1 >= cfg["min_drift"] and n0 >= cfg["min_non_drift"] and n_conv_drift >= cfg["min_drift"]
        row: dict = {
            "drift_next_pairs": n1,
            "normal_next_pairs": n0,
            "conversations_with_a_drift_next_pair": n_conv_drift,
            "enters": bool(enters),
            "pairs_by_band_drift_next_normal_next": [
                [int((drift_next[m] & (band[m] == b)).sum()), int((~drift_next[m] & (band[m] == b)).sum())]
                for b in range(len(BANDS))
            ],
        }
        if enters:
            xs, ys, bs, rs = x[first][m], drift_next[m], band[m], rating_k[m]
            n_conv = len(convs)
            w = resample_weights(n_conv, n_res, rng)

            def cells(groups: list[np.ndarray], values: np.ndarray = xs) -> list:
                return [pair_counts(values[g], ys[g], conv[g], n_conv) for g in groups if g.any()]

            by_band = [bs == b for b in range(len(BANDS))]
            by_band_rating = [(bs == b) & (rs == r) for b in range(len(BANDS)) for r in (0, 1)]
            banded[s] = cell_stats(cells(by_band), w, pct, lo, hi)
            unbanded[s] = cell_stats(cells([np.ones(len(ys), dtype=bool)]), w, pct, lo, hi)
            same_rating[s] = cell_stats(cells(by_band_rating), w, pct, lo, hi)
            rival = -rs[:, None].astype(float)  # lower means a higher rating at turn k
            row |= {
                "banded": public(banded[s]),
                "unbanded": public(unbanded[s]),
                "banded_same_rating_at_turn_k": public(same_rating[s]),
                "rival_rating_at_turn_k": {
                    name: {
                        "score": float(auc_cells(cells(groups, rival))[0] - 0.5),
                        "ci": [float(v - 0.5) for v in np.nanpercentile(auc_cells(cells(groups, rival), w)[:, 0], [lo, hi])],
                    }
                    for name, groups in (("banded", by_band), ("unbanded", [np.ones(len(ys), dtype=bool)]))
                },
            }
        out["strata"][s] = row
    out["n_strata_entering"] = len(banded)
    out["banded"] = average(banded, lo, hi)
    out["unbanded"] = average(unbanded, lo, hi)
    out["banded_same_rating_at_turn_k"] = average(same_rating, lo, hi)
    out["reading"] = lagged_reading(out)
    return out


def lagged_reading(out: dict) -> str:
    """The row of the table in section 16.2 that the result falls under."""
    if out["banded"] is None:
        return "Not run: fewer than two strata enter."
    if out["banded"]["above_zero"]:
        if out["banded_same_rating_at_turn_k"]["above_zero"]:
            return "Banded G_lag above zero, and still above zero among pairs with the same rating at turn k."
        return "Banded G_lag above zero, not above zero among pairs with the same rating at turn k."
    if out["unbanded"]["above_zero"]:
        return "Banded G_lag not above zero, unbanded above zero."
    return "G_lag not above zero either way."


def primary_reading(result: dict) -> str:
    """The row of the table in section 4.7 that the result falls under."""
    within, pooled = result["primary"], result["pooled"]
    if within is None:
        return "The primary estimand is not estimable: fewer than two strata enter."
    if within["above_zero"]:
        return "Gap above zero within strata."
    if pooled["above_zero"]:
        return "Gap above zero pooled, not within."
    return "Gap not above zero in either."


def describe(x: np.ndarray, drift: np.ndarray, percentile: float) -> dict:
    """AUC, bar, gap and rank on one set of turns, with no interval."""
    n1, n0 = int(drift.sum()), int((~drift).sum())
    out = {"n_drift": n1, "n_non_drift": n0}
    if n1 == 0 or n0 == 0:
        return out
    g = gap_from_auc(auc_cells([pair_counts(x, drift, np.zeros(len(drift), dtype=int), 1)]), percentile)
    return out | {
        "target_auc": float(g["target_score"] + 0.5),
        "bar": float(g["bar"]),
        "gap": float(g["gap"]),
        "rank": int(1 + (g["off_scores"] >= g["target_score"]).sum()),
    }


def human_labels(d: dict, x: np.ndarray, sample: list[dict], labels: dict[int, int], percentile: float, strata: list[str]) -> dict:
    """Section 15 (i). `labels` maps an item number to the owner's rating."""
    row_of = {key: i for i, key in enumerate(zip(d["conversation"].tolist(), d["turn"].tolist()))}
    items = [i for i in sample if i["item"] in labels]
    rows = np.array([row_of[(i["id"], i["turn"])] for i in items], dtype=int)
    owner = np.array([labels[i["item"]] for i in items])
    keep = d["has_activation"][rows] & (d["rating"][rows] >= 0)
    rows, owner = rows[keep], owner[keep]
    judge = d["rating"][rows]
    return {
        "turns": int(len(rows)),
        "left_out_no_activation_or_no_judge_rating": int((~keep).sum()),
        "owner": describe(x[rows], owner >= 2, percentile),
        "judge": describe(x[rows], judge >= 2, percentile),
        "counts_by_stratum": {
            s: {
                "turns": int((d["stratum"][rows] == s).sum()),
                "owner_drift": int(((owner >= 2) & (d["stratum"][rows] == s)).sum()),
                "judge_drift": int(((judge >= 2) & (d["stratum"][rows] == s)).sum()),
            }
            for s in strata
        },
    }


def sign_split(d: dict, x: np.ndarray, cfg: dict, rng: np.random.Generator) -> dict:
    """Section 15 (l). Drift turns with sign A (with or without C) and drift turns with C and
    not A, each against the same non-drift turns; drift turns of the other kind are left out."""
    lo, hi, pct, n_res = cfg["lo"], cfg["hi"], cfg["percentile"], cfg["n_resamples"]
    use = d["has_activation"] & (d["rating"] >= 0)
    drift = d["rating"] >= 2
    kinds = {"sign_a": drift & d["sign_a"], "sign_c_alone": drift & d["sign_c"] & ~d["sign_a"]}
    out: dict = {"strata": {}, "drift_turns_with_neither_sign": int((use & drift & ~d["sign_a"] & ~d["sign_c"]).sum())}
    entered: dict[str, dict] = {k: {} for k in kinds}
    for s in cfg["strata"]:
        m = use & (d["stratum"] == s)
        convs, conv = np.unique(d["conversation"][m], return_inverse=True)
        n_conv, xs, non = len(convs), x[m], ~drift[m]
        w = resample_weights(n_conv, n_res, rng)
        row: dict = {"n_non_drift": int(non.sum())}
        for name, kind in kinds.items():
            k = kind[m]
            enters = int(k.sum()) >= cfg["min_drift"] and int(non.sum()) >= cfg["min_non_drift"]
            row[name] = {"n_drift": int(k.sum()), "enters": bool(enters)}
            if enters:
                keep = k | non
                entered[name][s] = cell_stats([pair_counts(xs[keep], k[keep], conv[keep], n_conv)], w, pct, lo, hi)
                row[name] |= public(entered[name][s])
        if all(s in entered[name] for name in kinds):
            a, c = kinds["sign_a"][m], kinds["sign_c_alone"][m]
            keep = a | c
            between = [pair_counts(xs[keep], a[keep], conv[keep], n_conv)]
            g = gap_from_auc(auc_cells(between), pct)
            ci = np.nanpercentile(auc_cells(between, w)[:, 0], [lo, hi])
            diff = entered["sign_a"][s]["boot"] - entered["sign_c_alone"][s]["boot"]
            dci = np.nanpercentile(diff, [lo, hi])
            row["between_the_two_kinds"] = {
                "auc_sign_a_lower_than_c_alone": float(g["target_score"] + 0.5),
                "exploratory_ci": [float(ci[0]), float(ci[1])],
                "role_directions_p95_distance_from_half": float(g["bar"]),
            }
            row["exploratory_gap_sign_a_minus_c_alone"] = {
                "difference": float(entered["sign_a"][s]["gap"] - entered["sign_c_alone"][s]["gap"]),
                "ci_low": float(dci[0]),
                "ci_high": float(dci[1]),
            }
        out["strata"][s] = row
    for name in kinds:
        out[f"{name}_average_over_strata"] = average(entered[name], lo, hi)
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", required=True)
    args = parser.parse_args()
    with open(REPO_ROOT / "configs" / "study.toml", "rb") as f:
        study = tomllib.load(f)
    version = study["rubric"]["version"]
    strata = study["sample"]["strata"]
    level = study["bootstrap"]["ci_level"]
    cfg = {
        "strata": strata,
        "min_drift": study["sample"]["min_drift_turns"],
        "min_non_drift": study["sample"]["min_non_drift_turns"],
        "percentile": study["off_target"]["percentile"],
        "n_resamples": study["bootstrap"]["n_resamples"],
        "lo": 100 * (1 - level) / 2,
        "hi": 100 * (1 + level) / 2,
    }
    seed = study["seeds"]["bootstrap"]
    out_dir = REPO_ROOT / "results" / args.set
    proj_path, ratings_path = out_dir / "projections.npz", out_dir / f"ratings.rubric{version}.jsonl"
    z = np.load(proj_path)
    rating, sign_a, sign_c = join_ratings(
        z["conversation"], z["turn"], [json.loads(line) for line in ratings_path.read_text().splitlines() if line.strip()]
    )
    d = {
        "conversation": z["conversation"],
        "turn": z["turn"],
        "stratum": z["stratum"],
        "has_activation": z["has_activation"],
        "rating": rating,
        "sign_a": sign_a,
        "sign_c": sign_c,
    }
    layers = z["layers"].tolist()
    target_layer = json.loads((out_dir / "projection_log.json").read_text())["target_layer"]
    use = d["has_activation"] & (rating >= 0)

    def at(layer: int, off: str) -> np.ndarray:
        i = layers.index(layer)
        x = np.column_stack([z["axis"][:, i], z[off][:, i, :]]).astype(np.float64)
        if np.isnan(x[d["has_activation"]]).any():
            raise SystemExit("a turn marked as having an activation has no projection")
        return x

    def gap_test(x: np.ndarray, drift: np.ndarray, mask: np.ndarray, name: str, pooled: bool = True) -> dict:
        res = analyse(
            x[mask, 0], x[mask, 1:], drift[mask], d["conversation"][mask], d["stratum"][mask],
            min_drift=cfg["min_drift"], min_non_drift=cfg["min_non_drift"], percentile=cfg["percentile"],
            n_resamples=cfg["n_resamples"], ci_level=level, rng=stream(seed, name), pooled=pooled,
        )  # fmt: skip
        return res

    drift = rating >= 2
    x_roles = at(target_layer, "roles")
    primary = gap_test(x_roles, drift, use, "primary")
    primary["reading"] = primary_reading(primary)
    strict = rating == 3
    strict_res = gap_test(x_roles, strict, use, "strict_cut", pooled=int((strict & use).sum()) >= cfg["min_drift"])

    sample = json.loads((REPO_ROOT / "labels" / f"sample_{args.set}.json").read_text())["items"]
    final = latest_labels((REPO_ROOT / "labels" / labels_name(args.set, version)).read_text().splitlines())
    pilot = latest_labels((REPO_ROOT / "labels" / labels_name(args.set, 2)).read_text().splitlines())
    final_ratings = {n: lab["rating"] for n, lab in final.items()}
    all_ratings = {n: lab["rating"] for n, lab in pilot.items()} | final_ratings

    out = {
        "set": args.set,
        "inputs": {
            "projections_sha256": sha256(proj_path),
            "ratings": ratings_path.name,
            "ratings_sha256": sha256(ratings_path),
            "rubric_version": version,
            "target_layer": target_layer,
            "turns": int(len(rating)),
            "turns_used": int(use.sum()),
            "turns_without_activation": int((~d["has_activation"]).sum()),
            "turns_without_rating": int((rating < 0).sum()),
            "bootstrap_seed": seed,
            "n_resamples": cfg["n_resamples"],
        },
        "drift_counts": drift_counts(d["stratum"], d["turn"], rating, strata),
        "primary": {
            "role_directions": primary,
            "random_directions": gap_test(at(target_layer, "random"), drift, use, "primary_random"),
        },
        "layers": {
            str(layer): gap_test(at(layer, "roles"), drift, use, f"layer_{layer}")
            for layer in layers
            if layer != target_layer
        },
        "lagged": lagged(d, x_roles, cfg, stream(seed, "lagged")),
        "human_labels": {
            "items_41_to_120_version_3": human_labels(d, x_roles, sample, final_ratings, cfg["percentile"], strata),
            "all_120_mixing_versions_2_and_3": human_labels(d, x_roles, sample, all_ratings, cfg["percentile"], strata),
        },
        "strict_cut": strict_res,
        "sign_split": sign_split(d, x_roles, cfg, stream(seed, "sign_split")),
    }
    (out_dir / "analysis.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
