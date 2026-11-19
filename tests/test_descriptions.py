import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "analysis"))

import descriptions as ds  # noqa: E402
from gap import resample_weights  # noqa: E402

C = {"percentile": 95, "lo": 2.5, "hi": 97.5}


def data(seed=0, n_conv=20, turns=15, k=8):
    rng = np.random.default_rng(seed)
    conv = np.repeat(np.arange(n_conv), turns)
    turn = np.tile(np.arange(1, turns + 1), n_conv)
    rating = rng.choice(4, size=n_conv * turns, p=[0.3, 0.3, 0.25, 0.15])
    x = rng.standard_normal((n_conv * turns, k)).round(1)
    return x, rating, conv, turn, n_conv, rng


def test_within_conversation_auc_counts_only_pairs_from_one_conversation():
    x, rating, conv, _, n_conv, rng = data()
    drift = rating >= 2
    out = ds.item4_within_conversation(x, drift, conv, n_conv, resample_weights(n_conv, 50, rng), C)
    num = den = 0.0
    for c in range(n_conv):
        p, q = x[(conv == c) & drift, 0], x[(conv == c) & ~drift, 0]
        num += ((p[:, None] < q[None, :]) + 0.5 * (p[:, None] == q[None, :])).sum()
        den += len(p) * len(q)
    assert np.isclose(out["target_auc"], num / den) and out["pairs"] == den


def test_slope_removes_each_conversations_level():
    x, _, conv, turn, n_conv, _ = data()
    z = 0.3 * turn[:, None] + 10.0 * conv[:, None] + np.zeros((len(turn), 2))  # a level per conversation, one slope
    assert np.allclose(ds.within_slope(z, turn.astype(float), conv, n_conv), 0.3)
    w = np.ones((3, n_conv))
    assert np.allclose(ds.within_slope(z, turn.astype(float), conv, n_conv, w), 0.3)


def test_onsets_and_transitions():
    conversation = np.array(["a"] * 4 + ["b"] * 3 + ["c"] * 3)
    turn = np.array([1, 2, 3, 4, 1, 2, 3, 1, 2, 3])
    rating = np.array([0, 1, 2, 3, 2, 0, 0, 0, 1, 1])
    onset, from_first = ds.onsets(conversation, turn, rating)
    assert onset == {"a": 3} and from_first == 1
    out = ds.item2_transitions(conversation, turn, rating)
    assert out["conversations_with_a_drift_turn"] == 2 and out["of_which_drift_lasts_to_the_final_turn"] == 1
    table = np.array(out["table_rows_turn_k_cols_turn_k_plus_1"])
    assert table.sum() == 7 and table[0, 1] == 2 and table[2, 0] == 1 and table[2, 3] == 1


def test_variance_shares_add_to_one_and_rank_correlation():
    rng = np.random.default_rng(2)
    v = rng.standard_normal(60)
    stratum, conversation = np.repeat(["s", "t"], 30), np.repeat([f"c{i}" for i in range(6)], 10)
    assert np.isclose(sum(ds.variance_shares(v, stratum, conversation).values()), 1.0)
    assert np.isclose(ds.rank_corr(v, np.exp(v)), 1.0)


def test_signed_scores_and_top_roles():
    auc = np.array([0.7, 0.2, 0.55, 0.9])
    s = ds.signed(auc)
    assert np.isclose(s["axis_T"], 0.2) and s["roles_with_signed_score_above_axis_T"] == 1
    top = ds.top_roles(auc, ["p", "q", "r"])
    assert [t["role"] for t in top] == ["r", "p", "q"] and top[0]["side"] == "lower on drift" and top[1]["side"] == "higher on drift"
