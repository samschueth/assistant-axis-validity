"""The descriptions of PREREGISTRATION.md section 16.3, items 1 to 11.

    python src/analysis/descriptions.py --set main

Written on 2026-11-19, AFTER the registered tests (analyse_main.py) had been run and their
results seen (section 15). The items and what each reports were fixed in section 16.3 before any
rating existed; the code is later than the headline. Descriptions have no pass or fail.

Reads what analyse_main.py reads, and results/<set>/conversations.jsonl for reply lengths.
Writes results/<set>/descriptions.json. Calls no API.

Everything is within stratum, at the target layer unless a layer is named. A cell with fewer
than 15 turns (or pairs) is given as a count only. Intervals are 95%, from the conversation
bootstrap. "Roles" are the 275 role directions of section 4.3.

Choices section 16.3 leaves open:

- Item 3, slope: the projection is divided by that direction's standard deviation in the
  stratum; the slope is the least-squares slope on turn number with each conversation's own
  means removed (per turn, in standard deviations).
- Item 4: a resampled conversation brings its own within-conversation pairs with it; no pair is
  formed between two copies of one conversation.
- Item 5: curves are in standard deviations of each direction within the stratum, so that the
  axis and the roles can be set side by side; the axis is also given in its own units.
- Item 6: for the roles, the quantity shown is the share of drift-next pairs in the lowest third
  minus the share in the highest third.
- Item 7: "still normal at turn 5" means no drift turn among turns 1 to 5; the outcome is a
  drift turn among turns 6 to 8 (and likewise 10, and 11 to 13).
- Item 8: thirds of reply length are cut within the stratum.
- Items 9 to 11, "the secondary statistic" is the banded lagged test of section 16.2.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analyse_main as am  # noqa: E402
from gap import auc_cells, gap_from_auc, pair_counts, resample_weights  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MIN_CELL = 15
NAMED = {"emotional_disclosure": ["romantic", "familiar"], "meta_reflection": ["mystic", "spirit"]}


def pct(values: np.ndarray, lo: float, hi: float) -> list[float]:
    return [float(v) for v in np.nanpercentile(values, [lo, hi])]


def quantiles(values: np.ndarray) -> dict:
    return {str(q): float(np.percentile(values, q)) for q in (5, 25, 50, 75, 95)}


def weighted_mean(values: np.ndarray, conv: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The mean of `values` in each resample: turns weighted by their conversation's multiplicity."""
    tw = w[:, conv]
    den = tw.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (tw @ values) / np.where(den > 0, den, np.nan)


def gap_row(cells: list, w: np.ndarray, c: dict) -> dict:
    return am.public(am.cell_stats(cells, w, c["percentile"], c["lo"], c["hi"]))


def item1_ratings_in_order(x, rating, conv, n_conv, w, c) -> dict:
    out: dict = {"mean_projection_by_rating": {}, "adjacent_ratings": {}}
    for r in range(4):
        m = rating == r
        row: dict = {"turns": int(m.sum())}
        if m.sum() >= MIN_CELL:
            row |= {"mean": float(x[m, 0].mean()), "ci": pct(weighted_mean(x[m, 0], conv[m], w), c["lo"], c["hi"])}
        out["mean_projection_by_rating"][str(r)] = row
    for r in range(3):
        m = (rating == r) | (rating == r + 1)
        hi_n, lo_n = int((rating == r + 1).sum()), int((rating == r).sum())
        row = {"turns_lower_rating": lo_n, "turns_higher_rating": hi_n}
        if min(hi_n, lo_n) >= MIN_CELL:
            row |= gap_row([pair_counts(x[m], rating[m] == r + 1, conv[m], n_conv)], w, c)
        out["adjacent_ratings"][f"{r}_against_{r + 1}"] = row
    return out


def item2_transitions(conversation, turn, rating) -> dict:
    row_of = {key: i for i, key in enumerate(zip(conversation.tolist(), turn.tolist()))}
    table = np.zeros((4, 4), dtype=int)
    for (cid, t), i in row_of.items():
        j = row_of.get((cid, t + 1))
        if j is not None and rating[i] >= 0 and rating[j] >= 0:
            table[rating[i], rating[j]] += 1
    began, lasted = 0, 0
    for cid in dict.fromkeys(conversation.tolist()):
        m = conversation == cid
        r = rating[m][np.argsort(turn[m])]
        if (r >= 2).any():
            began += 1
            lasted += bool((r[np.argmax(r >= 2) :] >= 2).all())
    return {
        "table_rows_turn_k_cols_turn_k_plus_1": table.tolist(),
        "conversations_with_a_drift_turn": began,
        "of_which_drift_lasts_to_the_final_turn": lasted,
    }


def within_slope(z: np.ndarray, turn: np.ndarray, conv: np.ndarray, n_conv: int, w: np.ndarray | None = None) -> np.ndarray:
    """Slope of each column of z on turn number with each conversation's means removed."""
    onehot = np.zeros((n_conv, len(conv)))
    onehot[conv, np.arange(len(conv))] = 1.0
    counts = onehot.sum(axis=1)
    safe = np.where(counts > 0, counts, 1)
    t = turn - (onehot @ turn / safe)[conv]
    zc = z - (onehot @ z / safe[:, None])[conv]
    sxy, sxx = onehot @ (zc * t[:, None]), onehot @ (t**2)  # per conversation
    if w is None:
        return sxy.sum(axis=0) / sxx.sum()
    return (w @ sxy) / (w @ sxx)[:, None]


def item3_course(x, turn, conv, n_conv, w, c) -> dict:
    z = x / x.std(axis=0)
    slopes = within_slope(z, turn.astype(float), conv, n_conv)
    boot = within_slope(z[:, :1], turn.astype(float), conv, n_conv, w)[:, 0]
    return {
        "mean_projection_by_turn_number": {str(k): float(x[turn == k, 0].mean()) for k in sorted(set(turn.tolist()))},
        "axis_slope_sd_per_turn": float(slopes[0]),
        "axis_slope_ci": pct(boot, c["lo"], c["hi"]),
        "role_slopes_quantiles": quantiles(slopes[1:]),
        "axis_rank_by_absolute_slope": int(1 + (np.abs(slopes[1:]) >= abs(slopes[0])).sum()),
    }


def item4_within_conversation(x, drift, conv, n_conv, w, c) -> dict:
    pc = pair_counts(x, drift, conv, n_conv)
    diag = np.einsum("jcc->jc", pc.m)
    den = pc.n_drift * pc.n_non_drift
    row = {"pairs": int(den.sum()), "conversations_with_both_kinds": int((den > 0).sum())}
    if (den > 0).sum() < MIN_CELL:
        return row
    g = gap_from_auc(diag.sum(axis=1) / den.sum(), c["percentile"])
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = gap_from_auc((w @ diag.T) / np.where(w @ den > 0, w @ den, np.nan)[:, None], c["percentile"])["gap"]
    ci = pct(boot, c["lo"], c["hi"])
    return row | {
        "target_auc": float(g["target_score"] + 0.5),
        "bar": float(g["bar"]),
        "gap": float(g["gap"]),
        "rank": int(1 + (g["off_scores"] >= g["target_score"]).sum()),
        "ci_low": ci[0],
        "ci_high": ci[1],
    }


def onsets(conversation, turn, rating) -> tuple[dict, int]:
    """For each conversation with an onset, the turn number of the onset; and how many drift from turn 1."""
    out, from_first = {}, 0
    for cid in dict.fromkeys(conversation.tolist()):
        m = conversation == cid
        order = np.argsort(turn[m])
        r, t = rating[m][order], turn[m][order]
        if (r >= 2).any():
            k = int(np.argmax(r >= 2))
            if k == 0:
                from_first += 1
            else:
                out[cid] = int(t[k])
    return out, from_first


def item5_onset(x, conversation, turn, rating) -> dict:
    onset, from_first = onsets(conversation, turn, rating)
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    by_turn = {k: z[turn == k].mean(axis=0) for k in set(turn.tolist())}
    zd = z - np.stack([by_turn[k] for k in turn.tolist()])
    curves: dict = {}
    for off in range(-5, 6):
        rows = [
            i for i, (cid, t) in enumerate(zip(conversation.tolist(), turn.tolist())) if cid in onset and t - onset[cid] == off
        ]
        row: dict = {"conversations": len(rows)}
        if len(rows) >= MIN_CELL:
            a, b = z[rows].mean(axis=0), zd[rows].mean(axis=0)
            row |= {
                "axis_mean_in_its_own_units": float(x[rows, 0].mean()),
                "axis_sd_units": float(a[0]),
                "axis_sd_units_turn_mean_removed": float(b[0]),
                "roles_sd_units_5_to_95": [float(v) for v in np.percentile(a[1:], [5, 95])],
                "roles_sd_units_turn_mean_removed_5_to_95": [float(v) for v in np.percentile(b[1:], [5, 95])],
            }
        curves[str(off)] = row
    at = np.array(list(onset.values()), dtype=int)
    return {
        "conversations_with_an_onset": len(onset),
        "conversations_drifting_from_the_first_turn": from_first,
        "onsets_by_turn_number": {str(k): int((at == k).sum()) for k in sorted(set(at.tolist()))},
        "onsets_by_band_of_the_turn_before": [int(((at - 1 >= lo) & (at - 1 <= hi)).sum()) for lo, hi in am.BANDS],
        "turns_from_onset": curves,
    }


def item6_thirds(xk, drift_next) -> dict:
    def shares(v: np.ndarray) -> list:
        cuts = np.quantile(v, [1 / 3, 2 / 3])
        third = np.digitize(v, cuts)
        return [[float(drift_next[third == k].mean()), int((third == k).sum())] for k in range(3)]

    if len(drift_next) < 3 * MIN_CELL:
        return {"pairs": int(len(drift_next))}
    axis = shares(xk[:, 0])
    roles = np.array([(lambda s: s[0][0] - s[2][0])(shares(xk[:, j])) for j in range(1, xk.shape[1])])
    return {
        "pairs": int(len(drift_next)),
        "axis_share_drift_next_and_pairs_lowest_middle_highest_third": axis,
        "axis_lowest_minus_highest": float(axis[0][0] - axis[2][0]),
        "roles_lowest_minus_highest_quantiles": quantiles(roles),
        "axis_rank_by_absolute_difference": int(1 + (np.abs(roles) >= abs(axis[0][0] - axis[2][0])).sum()),
    }


def item7_turning_points(x, conversation, turn, rating, has, percentile) -> dict:
    out = {}
    for point in (5, 10):
        rows, later = [], []
        for cid in dict.fromkeys(conversation.tolist()):
            m = conversation == cid
            r = dict(zip(turn[m].tolist(), rating[m].tolist()))
            i = np.flatnonzero(m & (turn == point))
            if len(i) == 0 or not has[i[0]] or any(r.get(k, 0) >= 2 for k in range(1, point + 1)):
                continue
            if not all(k in r for k in range(point + 1, point + 4)):
                continue
            rows.append(int(i[0]))
            later.append(any(r[k] >= 2 for k in range(point + 1, point + 4)))
        later_a = np.array(later, dtype=bool)
        row: dict = {
            "conversations_still_normal": len(rows),
            "drift_in_the_next_three_turns": int(later_a.sum()),
            "no_drift_in_the_next_three_turns": int((~later_a).sum()),
        }
        if min(later_a.sum(), (~later_a).sum()) >= MIN_CELL:
            row |= am.describe(x[rows], later_a, percentile)
        out[f"turn_{point}"] = row
    return out


def variance_shares(v: np.ndarray, stratum: np.ndarray, conversation: np.ndarray) -> dict:
    grand = v.mean()
    total = ((v - grand) ** 2).sum()
    s_mean = {s: v[stratum == s].mean() for s in set(stratum.tolist())}
    c_mean = {cid: v[conversation == cid].mean() for cid in set(conversation.tolist())}
    sm = np.array([s_mean[s] for s in stratum.tolist()])
    cm = np.array([c_mean[cid] for cid in conversation.tolist()])
    return {
        "between_strata": float(((sm - grand) ** 2).sum() / total),
        "between_conversations_within_stratum": float(((cm - sm) ** 2).sum() / total),
        "within_conversations": float(((v - cm) ** 2).sum() / total),
    }


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        return (np.bincount(inv, weights=r) / counts)[inv]

    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def top_roles(auc: np.ndarray, names: list[str]) -> list[dict]:
    """The ten role directions furthest from 0.5. `auc` is the AUC for a LOWER projection on drift."""
    score = np.abs(auc[1:] - 0.5)
    return [
        {"role": names[j], "score": float(score[j]), "side": "lower on drift" if auc[1 + j] > 0.5 else "higher on drift"}
        for j in np.argsort(-score)[:10]
    ]


def signed(auc: np.ndarray) -> dict:
    """Item 10: every role direction scored for a HIGHER projection on drift."""
    s = 0.5 - auc[1:]
    return {
        "axis_T": float(auc[0] - 0.5),
        "role_signed_score_quantiles": quantiles(s),
        "share_of_roles_above_zero": float((s > 0).mean()),
        "roles_with_signed_score_above_axis_T": int((s >= auc[0] - 0.5).sum()),
    }


def named(cells: list, w: np.ndarray, names: list[str], which: list[str], c: dict) -> dict:
    """Item 11: each named direction's signed score, and its difference from the axis's T."""
    auc, boot = auc_cells(cells), auc_cells(cells, w)
    out = {}
    for role in which:
        j = 1 + names.index(role)
        diff = (0.5 - boot[:, j]) - (boot[:, 0] - 0.5)
        ci = pct(diff, c["lo"], c["hi"])
        out[role] = {
            "signed_score": float(0.5 - auc[j]),
            "axis_T": float(auc[0] - 0.5),
            "difference_from_axis_T": float((0.5 - auc[j]) - (auc[0] - 0.5)),
            "ci_low": ci[0],
            "ci_high": ci[1],
            "above_zero": bool(ci[0] > 0),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", required=True)
    args = parser.parse_args()
    with open(REPO_ROOT / "configs" / "study.toml", "rb") as f:
        study = tomllib.load(f)
    level = study["bootstrap"]["ci_level"]
    c = {
        "strata": study["sample"]["strata"],
        "min_drift": study["sample"]["min_drift_turns"],
        "min_non_drift": study["sample"]["min_non_drift_turns"],
        "percentile": study["off_target"]["percentile"],
        "n_resamples": study["bootstrap"]["n_resamples"],
        "lo": 100 * (1 - level) / 2,
        "hi": 100 * (1 + level) / 2,
    }
    seed = study["seeds"]["bootstrap"]
    out_dir = REPO_ROOT / "results" / args.set
    z = np.load(out_dir / "projections.npz")
    ratings_path = out_dir / f"ratings.rubric{study['rubric']['version']}.jsonl"
    rating, sign_a, sign_c = am.join_ratings(
        z["conversation"], z["turn"], [json.loads(line) for line in ratings_path.read_text().splitlines() if line.strip()]
    )
    conversation, turn, stratum, has = z["conversation"], z["turn"], z["stratum"], z["has_activation"]
    names = z["role_names"].tolist()
    layers = z["layers"].tolist()
    target = json.loads((out_dir / "projection_log.json").read_text())["target_layer"]
    tokens = {
        (cv["id"], k): t["reply_tokens"]
        for cv in map(json.loads, (out_dir / "conversations.jsonl").read_text().splitlines())
        for k, t in enumerate(cv["turns"], start=1)
    }
    length = np.array([tokens[key] for key in zip(conversation.tolist(), turn.tolist())], dtype=float)

    def at(layer: int) -> np.ndarray:
        i = layers.index(layer)
        return np.column_stack([z["axis"][:, i], z["roles"][:, i, :]]).astype(np.float64)

    x_all = at(target)
    use = has & (rating >= 0)
    d = {"conversation": conversation, "turn": turn, "stratum": stratum, "has_activation": has, "rating": rating,
         "sign_a": sign_a, "sign_c": sign_c}  # fmt: skip
    first, second = am.lagged_pairs(conversation, turn, rating, has)

    out: dict = {
        "set": args.set,
        "note": "Coded after the registered tests were run and seen (PREREGISTRATION.md section 15, 2026-11-19).",
        "target_layer": target,
        "strata": {},
        "item8_noise_all_strata": {
            "variance_shares_of_the_axis_projection": variance_shares(x_all[use, 0], stratum[use], conversation[use]),
            "lagged_test_at_reliability_layers": {
                str(layer): {
                    k: v
                    for k, v in am.lagged(d, at(layer), c, am.stream(seed, f"descriptions_lagged_{layer}")).items()
                    if k != "strata"
                }
                for layer in layers
                if layer != target
            },
        },
    }
    for s in c["strata"]:
        rng = am.stream(seed, f"descriptions_{s}")
        m = use & (stratum == s)
        convs, conv = np.unique(conversation[m], return_inverse=True)
        n_conv = len(convs)
        w = resample_weights(n_conv, c["n_resamples"], rng)
        x, r, t = x_all[m], rating[m], turn[m]
        drift = r >= 2
        enters = drift.sum() >= c["min_drift"] and (~drift).sum() >= c["min_non_drift"]
        row: dict = {
            "item1_ratings_in_order": item1_ratings_in_order(x, r, conv, n_conv, w, c),
            "item2_ratings_from_one_turn_to_the_next": item2_transitions(
                conversation[stratum == s], turn[stratum == s], rating[stratum == s]
            ),
            "item3_course_over_turns": item3_course(x, t, conv, n_conv, w, c),
            "item5_around_the_onset": item5_onset(x, conversation[m], t, r),
            "item7_turning_points": item7_turning_points(
                x_all[stratum == s], conversation[stratum == s], turn[stratum == s], rating[stratum == s],
                has[stratum == s], c["percentile"],
            ),  # fmt: skip
            "item8_noise": {"rank_correlation_of_axis_projection_with_reply_tokens": rank_corr(x[:, 0], length[m])},
        }
        # lagged pairs of this stratum, on the conversation index of its used turns
        pm = stratum[first] == s
        idx = {cid: i for i, cid in enumerate(convs.tolist())}
        pconv = np.array([idx[cid] for cid in conversation[first][pm].tolist()], dtype=int)
        xk, nxt, band = x_all[first][pm], rating[second][pm] >= 2, am.band_of(turn[first][pm])
        row["item6_onset_rate_by_projection_level"] = item6_thirds(xk, nxt)
        lag_enters = (
            nxt.sum() >= c["min_drift"] and (~nxt).sum() >= c["min_non_drift"] and len(set(pconv[nxt].tolist())) >= c["min_drift"]
        )
        if enters:
            primary_cells = [pair_counts(x, drift, conv, n_conv)]
            auc = auc_cells(primary_cells)
            row["item4_within_one_conversation"] = item4_within_conversation(x, drift, conv, n_conv, w, c)
            third = np.digitize(length[m], np.quantile(length[m], [1 / 3, 2 / 3]))
            row["item8_noise"]["primary_within_thirds_of_reply_length"] = gap_row(
                [pair_counts(x[third == k], drift[third == k], conv[third == k], n_conv) for k in range(3)], w, c
            )
            med_non, med_drift = np.median(x[~drift, 0]), np.median(x[drift, 0])
            row["item9_what_the_axis_leaves_out"] = {
                "primary_axis_rank_of_276": int(1 + (np.abs(auc[1:] - 0.5) >= auc[0] - 0.5).sum()),
                "primary_top_ten_roles": top_roles(auc, names),
                "share_of_drift_turns_above_the_non_drift_median": float((x[drift, 0] > med_non).mean()),
                "share_of_non_drift_turns_below_the_drift_median": float((x[~drift, 0] < med_drift).mean()),
            }
            row["item10_roles_scored_on_one_side"] = {"primary": signed(auc)}
            if s in NAMED:
                row["item11_named_directions"] = {"primary": named(primary_cells, w, names, NAMED[s], c)}
        if lag_enters:
            lag_cells = [pair_counts(xk[band == b], nxt[band == b], pconv[band == b], n_conv) for b in range(len(am.BANDS))]
            lag_cells = [pc for pc in lag_cells if pc.n_drift.sum() + pc.n_non_drift.sum() > 0]
            lauc = auc_cells(lag_cells)
            row.setdefault("item9_what_the_axis_leaves_out", {}).update(
                {
                    "secondary_axis_rank_of_276": int(1 + (np.abs(lauc[1:] - 0.5) >= lauc[0] - 0.5).sum()),
                    "secondary_top_ten_roles": top_roles(lauc, names),
                }
            )
            row.setdefault("item10_roles_scored_on_one_side", {})["secondary"] = signed(lauc)
            if s in NAMED:
                row.setdefault("item11_named_directions", {})["secondary"] = named(lag_cells, w, names, NAMED[s], c)
        out["strata"][s] = row
    (out_dir / "descriptions.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
