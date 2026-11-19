import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "analysis"))

import analyse_main as am  # noqa: E402
from gap import auc_cells, pair_counts  # noqa: E402

CFG = {"strata": ["s1", "s2"], "min_drift": 15, "min_non_drift": 15, "percentile": 95, "n_resamples": 200, "lo": 2.5, "hi": 97.5}


def brute(p, q):
    return ((p[:, None] < q[None, :]) + 0.5 * (p[:, None] == q[None, :])).sum(), len(p) * len(q)


def synthetic(seed=0, n_conv=24, turns=15, k=12, effect=1.0):
    rng = np.random.default_rng(seed)
    n = n_conv * turns
    conversation = np.repeat([f"c{i:02d}" for i in range(n_conv)], turns)
    turn = np.tile(np.arange(1, turns + 1), n_conv)
    stratum = np.repeat(["s1", "s2"], n // 2)
    rating = rng.choice(4, size=n, p=[0.3, 0.3, 0.25, 0.15])
    sign_a = (rating >= 2) & (rng.random(n) < 0.5)
    sign_c = (rating >= 2) & ~sign_a
    x = rng.standard_normal((n, k)).round(1)
    x[:, 0] -= effect * (rating >= 2)
    d = {
        "conversation": conversation, "turn": turn, "stratum": stratum, "rating": rating,
        "has_activation": np.ones(n, dtype=bool), "sign_a": sign_a, "sign_c": sign_c,
    }  # fmt: skip
    return d, x, rng


def test_join_is_by_key_not_by_row_order():
    conversation, turn = np.array(["a", "a", "b"]), np.array([1, 2, 1])
    rows = [
        {"id": "b", "turn": 1, "rating": 3, "signs": ["A", "C"]},
        {"id": "a", "turn": 2, "rating": 2, "signs": ["C"]},
        {"id": "a", "turn": 1, "rating": 0, "signs": []},
    ]
    rating, a, c = am.join_ratings(conversation, turn, rows)
    assert rating.tolist() == [0, 2, 3] and a.tolist() == [False, False, True] and c.tolist() == [False, True, True]
    with pytest.raises(SystemExit):
        am.join_ratings(conversation, turn, rows[:2])
    with pytest.raises(SystemExit):
        am.join_ratings(conversation, turn, rows + [rows[0]])


def test_lagged_pairs_follow_section_16_1():
    conversation = np.array(["a"] * 4 + ["b"] * 2)
    turn = np.array([1, 2, 3, 4, 1, 2])
    rating = np.array([0, 1, 2, 0, 1, 3])
    has = np.array([True, False, True, True, True, True])
    first, second = am.lagged_pairs(conversation, turn, rating, has)
    # a1->a2 (normal, has activation); a2 has no activation; a3 is a drift turn; a4 is last; b1->b2
    assert list(zip(first.tolist(), second.tolist())) == [(0, 1), (4, 5)]


def test_banded_auc_adds_counts_over_cells_before_dividing():
    rng = np.random.default_rng(1)
    n, n_conv = 200, 10
    x = rng.standard_normal((n, 3)).round(1)
    y = rng.random(n) < 0.4
    conv = rng.integers(0, n_conv, n)
    band = rng.integers(0, 3, n)
    cells = [pair_counts(x[band == b], y[band == b], conv[band == b], n_conv) for b in range(3)]
    got = auc_cells(cells)
    for j in range(3):
        parts = [brute(x[(band == b) & y, j], x[(band == b) & ~y, j]) for b in range(3)]
        assert np.isclose(got[j], sum(p[0] for p in parts) / sum(p[1] for p in parts))
    # a weight of 2 on a conversation is the same as having that conversation twice
    w = np.ones((1, n_conv))
    assert np.allclose(auc_cells(cells, w)[0], got)
    w[0, 3] = 2
    dup = conv == 3
    x2, y2, band2 = np.vstack([x, x[dup]]), np.concatenate([y, y[dup]]), np.concatenate([band, band[dup]])
    conv2 = np.concatenate([conv, np.full(dup.sum(), n_conv)])
    cells2 = [pair_counts(x2[band2 == b], y2[band2 == b], conv2[band2 == b], n_conv + 1) for b in range(3)]
    assert np.allclose(auc_cells(cells, w)[0], auc_cells(cells2))


def test_sign_split_leaves_out_the_other_kind_of_drift_turn():
    d, x, rng = synthetic()
    c_alone = (d["rating"] >= 2) & d["sign_c"] & ~d["sign_a"]
    x[c_alone, 0] = 99.0  # would wreck the sign A figure if these turns were counted as non-drift
    out = am.sign_split(d, x, CFG, rng)
    for s in CFG["strata"]:
        m = d["stratum"] == s
        a = m & (d["rating"] >= 2) & d["sign_a"]
        num, den = brute(x[a, 0], x[m & (d["rating"] <= 1), 0])
        assert np.isclose(out["strata"][s]["sign_a"]["target_auc"], num / den)
        num, den = brute(x[a, 0], x[m & c_alone, 0])
        assert np.isclose(out["strata"][s]["between_the_two_kinds"]["auc_sign_a_lower_than_c_alone"], num / den)
    assert out["sign_a_average_over_strata"]["strata"] == ["s1", "s2"]


def test_lagged_test_matches_brute_force_and_reads_a_result():
    d, x, rng = synthetic(seed=3, n_conv=40)
    out = am.lagged(d, x, CFG, rng)
    first, second = am.lagged_pairs(d["conversation"], d["turn"], d["rating"], d["has_activation"])
    s = "s1"
    m = d["stratum"][first] == s
    y, band = d["rating"][second][m] >= 2, am.band_of(d["turn"][first][m])
    v = x[first][m, 0]
    parts = [brute(v[(band == b) & y], v[(band == b) & ~y]) for b in range(3)]
    assert np.isclose(out["strata"][s]["banded"]["target_auc"], sum(p[0] for p in parts) / sum(p[1] for p in parts))
    num, den = brute(v[y], v[~y])
    assert np.isclose(out["strata"][s]["unbanded"]["target_auc"], num / den)
    r = -d["rating"][first][m].astype(float)
    num, den = brute(r[y], r[~y])
    assert np.isclose(out["strata"][s]["rival_rating_at_turn_k"]["unbanded"]["score"], num / den - 0.5)
    assert out["reading"] in {
        "Banded G_lag above zero, and still above zero among pairs with the same rating at turn k.",
        "Banded G_lag above zero, not above zero among pairs with the same rating at turn k.",
        "Banded G_lag not above zero, unbanded above zero.",
        "G_lag not above zero either way.",
    }


def test_human_label_description_uses_the_sampled_turns_only():
    d, x, _ = synthetic(seed=5)
    sample = [{"item": n + 1, "id": d["conversation"][i], "turn": int(d["turn"][i])} for n, i in enumerate(range(0, 360, 7))]
    labels = {i["item"]: int(n % 4) for n, i in enumerate(sample)}
    out = am.human_labels(d, x, sample, labels, 95, CFG["strata"])
    rows = np.arange(0, 360, 7)
    owner = np.array([n % 4 for n in range(len(rows))]) >= 2
    num, den = brute(x[rows][owner, 0], x[rows][~owner, 0])
    assert out["turns"] == len(rows) and np.isclose(out["owner"]["target_auc"], num / den)
    judge = d["rating"][rows] >= 2
    num, den = brute(x[rows][judge, 0], x[rows][~judge, 0])
    assert np.isclose(out["judge"]["target_auc"], num / den)


def test_readings_of_section_4_7():
    assert am.primary_reading({"primary": None, "pooled": {"above_zero": True}}).startswith("The primary estimand is not estimable")
    assert am.primary_reading({"primary": {"above_zero": True}, "pooled": {"above_zero": False}}) == "Gap above zero within strata."
    assert am.primary_reading({"primary": {"above_zero": False}, "pooled": {"above_zero": True}}) == "Gap above zero pooled, not within."
    assert am.primary_reading({"primary": {"above_zero": False}, "pooled": {"above_zero": False}}) == "Gap not above zero in either."
