"""The gap statistic of PREREGISTRATION.md section 4, with its conversation-level bootstrap.

For any direction, the AUC is the probability that a randomly chosen drift turn has a LOWER
projection than a randomly chosen non-drift turn, with ties counted as one half.

    target score      T   = AUC(axis) - 0.5              one-sided
    off-target score  O_j = |AUC(direction j) - 0.5|     two-sided
    gap               G   = T - P95(O)

The bootstrap resamples conversations, not turns. An AUC is a ratio of pair counts, and the
pair counts add up over pairs of conversations, so they are computed once per stratum and
every resample is then a reweighting of them. This is exact, not an approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CHUNK = 48  # directions per block when counting pairs, to bound memory
RESAMPLE_CHUNK = 250  # resamples per block when reweighting, to bound memory


@dataclass(frozen=True)
class PairCounts:
    """Pair counts for one set of turns, grouped by conversation.

    m[j, c, d] is the number of (drift turn in conversation c, non-drift turn in conversation d)
    pairs in which the drift turn has the lower projection on direction j, ties as one half.
    """

    m: np.ndarray  # (n_directions, n_conversations, n_conversations)
    n_drift: np.ndarray  # (n_conversations,)
    n_non_drift: np.ndarray  # (n_conversations,)


def pair_counts(x: np.ndarray, drift: np.ndarray, conversation: np.ndarray, n_conv: int | None = None) -> PairCounts:
    """Count ordered pairs. x is (n_turns, n_directions); drift is boolean (n_turns,).

    With `n_conv`, `conversation` holds integer indices from 0 to n_conv - 1 that are kept as
    they are, so that counts made on different subsets of the turns share one conversation
    index and can be reweighted by the same bootstrap draw."""
    x = np.asarray(x, dtype=np.float64)
    drift = np.asarray(drift, dtype=bool)
    if n_conv is None:
        _, conv = np.unique(conversation, return_inverse=True)
        n_conv = int(conv.max()) + 1
    else:
        conv = np.asarray(conversation, dtype=int)
    onehot = np.zeros((n_conv, len(conv)), dtype=np.float32)
    onehot[conv, np.arange(len(conv))] = 1.0
    a = onehot[:, drift]  # (C, n_drift_turns)
    b = onehot[:, ~drift]  # (C, n_non_drift_turns)
    xp, xq = x[drift], x[~drift]
    blocks = []
    for start in range(0, x.shape[1], CHUNK):
        p = xp[:, None, start : start + CHUNK]
        q = xq[None, :, start : start + CHUNK]
        k = (p < q).astype(np.float32) + 0.5 * (p == q).astype(np.float32)
        ak = np.tensordot(a, k, axes=(1, 0))  # (C, n_non_drift_turns, block)
        blocks.append(np.tensordot(ak, b, axes=(1, 1)).transpose(1, 0, 2))  # (block, C, C)
    m = np.concatenate(blocks, axis=0).astype(np.float64)
    return PairCounts(m=m, n_drift=a.sum(axis=1).astype(np.float64), n_non_drift=b.sum(axis=1).astype(np.float64))


def auc_cells(cells: list[PairCounts], weights: np.ndarray | None = None) -> np.ndarray:
    """AUC per direction when only pairs inside the same cell are compared: the concordant
    counts and the pair counts are added over the cells before dividing. The cells must share
    one conversation index. With weights of shape (n_resamples, n_conversations), one row per
    resample; a resample with no pair to compare gives NaN."""
    if weights is None:
        num = sum(pc.m.sum(axis=(1, 2)) for pc in cells)
        den = sum(pc.n_drift.sum() * pc.n_non_drift.sum() for pc in cells)
        return num / den if den > 0 else np.full(cells[0].m.shape[0], np.nan)
    w_all = np.asarray(weights, dtype=np.float64)
    rows = []
    for start in range(0, len(w_all), RESAMPLE_CHUNK):
        w = w_all[start : start + RESAMPLE_CHUNK]
        num, den = 0.0, 0.0
        for pc in cells:
            wm = np.tensordot(w, pc.m, axes=(1, 1))  # (B, n_directions, C)
            num = num + (wm * w[:, None, :]).sum(axis=2)
            den = den + (w @ pc.n_drift) * (w @ pc.n_non_drift)
        with np.errstate(invalid="ignore", divide="ignore"):
            rows.append(num / np.where(den > 0, den, np.nan)[:, None])
    return np.concatenate(rows, axis=0)


def auc_lower(pc: PairCounts, weights: np.ndarray | None = None) -> np.ndarray:
    """AUC per direction. With weights of shape (n_resamples, n_conversations), one row per
    resample; a resample with no drift turn or no non-drift turn gives NaN."""
    return auc_cells([pc], weights)


def gap_from_auc(auc: np.ndarray, percentile: float = 95.0) -> dict[str, np.ndarray]:
    """Column 0 of the last axis is the target; the rest are off-target directions."""
    t = auc[..., 0] - 0.5
    o = np.abs(auc[..., 1:] - 0.5)
    bar = np.percentile(o, percentile, axis=-1)
    return {"target_score": t, "bar": bar, "gap": t - bar, "off_scores": o}


def resample_weights(n_conv: int, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    """Multiplicities from drawing n_conv conversations with replacement, per resample."""
    return rng.multinomial(n_conv, np.full(n_conv, 1.0 / n_conv), size=n_resamples).astype(np.float64)


def analyse(
    target: np.ndarray,
    off_target: np.ndarray,
    drift: np.ndarray,
    conversation: np.ndarray,
    stratum: np.ndarray,
    *,
    min_drift: int = 15,
    min_non_drift: int = 15,
    percentile: float = 95.0,
    n_resamples: int = 2000,
    ci_level: float = 0.95,
    rng: np.random.Generator,
    pooled: bool = True,
) -> dict:
    """Within-stratum (primary) and pooled (secondary) gap, with bootstrap intervals."""
    x = np.column_stack([target, off_target])
    drift = np.asarray(drift, dtype=bool)
    lo, hi = 100 * (1 - ci_level) / 2, 100 * (1 + ci_level) / 2
    out: dict = {"strata": {}}

    entering, boot_gaps = [], []
    for s in sorted(set(stratum.tolist())):
        mask = stratum == s
        n1, n0 = int(drift[mask].sum()), int((~drift[mask]).sum())
        enters = n1 >= min_drift and n0 >= min_non_drift
        row = {"n_drift": n1, "n_non_drift": n0, "enters": enters}
        if enters:
            pc = pair_counts(x[mask], drift[mask], conversation[mask])
            g = gap_from_auc(auc_lower(pc), percentile)
            row |= {
                "target_auc": float(g["target_score"] + 0.5),
                "bar": float(g["bar"]),
                "gap": float(g["gap"]),
                "rank": int(1 + (g["off_scores"] >= g["target_score"]).sum()),
                "n_off_target": int(g["off_scores"].size),
                "off_score_quantiles": {
                    str(q): float(np.percentile(g["off_scores"], q)) for q in (5, 25, 50, 75, 90, 95, 99, 100)
                },
            }
            w = resample_weights(len(pc.n_drift), n_resamples, rng)
            boot_gaps.append(gap_from_auc(auc_lower(pc, w), percentile)["gap"])
            s_ci = np.nanpercentile(boot_gaps[-1], [lo, hi])  # this stratum alone, from the same draws
            row |= {"ci_low": float(s_ci[0]), "ci_high": float(s_ci[1])}
            entering.append(s)
        out["strata"][s] = row

    out["n_strata_entering"] = len(entering)
    if len(entering) >= 2:
        point = float(np.mean([out["strata"][s]["gap"] for s in entering]))
        # A resample in which a stratum has an empty class drops that stratum for that draw.
        boot = np.nanmean(np.vstack(boot_gaps), axis=0)
        ci = np.nanpercentile(boot, [lo, hi])
        out["primary"] = {"gap": point, "ci_low": float(ci[0]), "ci_high": float(ci[1]), "above_zero": bool(ci[0] > 0)}
    else:
        out["primary"] = None  # not estimable: fewer than two strata enter

    if pooled:
        pc = pair_counts(x, drift, conversation)
        g = gap_from_auc(auc_lower(pc), percentile)
        # Resample conversations within stratum, as for the primary estimand.
        _, conv = np.unique(conversation, return_inverse=True)
        conv_stratum = np.empty(conv.max() + 1, dtype=stratum.dtype)
        conv_stratum[conv] = stratum
        w = np.zeros((n_resamples, len(conv_stratum)))
        for s in sorted(set(stratum.tolist())):
            idx = np.flatnonzero(conv_stratum == s)
            w[:, idx] = resample_weights(len(idx), n_resamples, rng)
        ci = np.nanpercentile(gap_from_auc(auc_lower(pc, w), percentile)["gap"], [lo, hi])
        out["pooled"] = {
            "target_auc": float(g["target_score"] + 0.5),
            "bar": float(g["bar"]),
            "gap": float(g["gap"]),
            "rank": int(1 + (g["off_scores"] >= g["target_score"]).sum()),
            "ci_low": float(ci[0]),
            "ci_high": float(ci[1]),
            "above_zero": bool(ci[0] > 0),
        }
    return out
