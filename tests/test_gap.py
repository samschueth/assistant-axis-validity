import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "analysis"))

from gap import analyse, auc_lower, gap_from_auc, pair_counts  # noqa: E402


def brute_auc_lower(x, drift):
    p, q = x[drift], x[~drift]
    return ((p[:, None] < q[None, :]) + 0.5 * (p[:, None] == q[None, :])).mean()


def make(n_conv=12, turns=10, k=7, seed=0):
    rng = np.random.default_rng(seed)
    conv = np.repeat(np.arange(n_conv), turns)
    drift = rng.random(n_conv * turns) < 0.4
    x = rng.standard_normal((n_conv * turns, k)).round(1)  # rounding creates ties
    return x, drift, conv, rng


def test_auc_matches_brute_force_with_ties():
    x, drift, conv, _ = make()
    got = auc_lower(pair_counts(x, drift, conv))
    want = [brute_auc_lower(x[:, j], drift) for j in range(x.shape[1])]
    assert np.allclose(got, want)


def test_auc_direction_lower_projection_means_drift():
    drift = np.array([True] * 5 + [False] * 5)
    x = np.where(drift, -1.0, 1.0)[:, None]
    assert auc_lower(pair_counts(x, drift, np.arange(10)))[0] == 1.0


def test_reweighting_equals_recomputing_on_the_resampled_conversations():
    x, drift, conv, rng = make()
    pc = pair_counts(x, drift, conv)
    w = rng.multinomial(12, np.full(12, 1 / 12), size=1).astype(float)
    got = auc_lower(pc, w)[0]
    rows = np.concatenate([np.flatnonzero(conv == c) for c in range(12) for _ in range(int(w[0, c]))])
    want = [brute_auc_lower(x[rows, j], drift[rows]) for j in range(x.shape[1])]
    assert np.allclose(got, want)


def test_gap_scales_target_one_sided_and_off_target_two_sided():
    auc = np.array([0.40, 0.30, 0.70, 0.50])  # target below 0.5; off-targets 0.2, 0.2, 0.0 away
    g = gap_from_auc(auc, percentile=100)
    assert np.isclose(g["target_score"], -0.10)
    assert np.isclose(g["bar"], 0.20)
    assert np.isclose(g["gap"], -0.30)


def test_stratum_entry_rule_and_not_estimable():
    x, drift, conv, rng = make(n_conv=12, turns=10)
    stratum = np.where(conv < 6, "a", "b")
    drift = drift.copy()
    drift[stratum == "b"] = False  # stratum b has no drift turns
    res = analyse(x[:, 0], x[:, 1:], drift, conv, stratum, n_resamples=50, rng=rng)
    assert res["strata"]["b"]["enters"] is False
    assert res["n_strata_entering"] == 1
    assert res["primary"] is None
    assert "gap" in res["pooled"]
