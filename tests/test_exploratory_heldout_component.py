import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "analysis"))

import exploratory_heldout_component as hc  # noqa: E402
from gap import resample_weights  # noqa: E402

CFG = {"percentile": 95, "lo": 2.5, "hi": 97.5}


def planted(seed: int, extra: float, n_conv: int = 30, turns: int = 15, d: int = 40):
    """Activations with one strong direction that falls with the turn number, plus `extra` times
    a per-turn quantity that also raises the chance of drift at a fixed turn number."""
    rng = np.random.default_rng(seed)
    conv = np.repeat(np.arange(n_conv), turns)
    turn = np.tile(np.arange(1, turns + 1), n_conv)
    push = rng.standard_normal(n_conv * turns)
    drift = (turn - 8) / 3 + push + 0.3 * rng.standard_normal(n_conv * turns) > 0
    ref = np.zeros(d)
    ref[0] = 1.0
    along = -(turn - 8.0) - extra * 3.0 * push
    acts = 5.0 + np.outer(along, ref) + 0.5 * rng.standard_normal((n_conv * turns, d))
    others = np.column_stack([acts @ ref, rng.standard_normal((n_conv * turns, 6))])  # axis, 3 roles, 3 random
    return acts, others, drift, conv, turn, ref, rng


def run(seed: int, extra: float) -> dict:
    acts, others, drift, conv, turn, ref, rng = planted(seed, extra)
    w = resample_weights(30, 200, rng)
    return hc.analyse_stratum(acts, others, drift, conv, turn, ref, 3, w, np.arange(30) % hc.N_FOLDS, CFG)


def test_component_that_is_only_the_turn_number_does_not_separate_at_a_fixed_turn():
    res = run(0, extra=0.0)
    assert res["two_by_two"]["heldout_unmatched"]["component_auc"] > 0.75
    lo, hi = res["headline_heldout_matched"]["component_auc_ci"]
    assert lo < 0.5 < hi


def test_component_that_carries_more_than_the_turn_number_does():
    res = run(1, extra=1.0)
    assert res["headline_heldout_matched"]["component_auc_ci"][0] > 0.5
    assert res["five_fold"]["matched_within_fold"]["component_auc"] > 0.6


def test_fits_share_one_orientation_and_matched_pairs_add_up():
    res = run(2, extra=1.0)
    assert res["fits"]["min_cosine_between_fits"] > 0.9 and res["fits"]["cosine_with_axis_fit_on_all"] > 0
    pairs = sum(v["pairs"] for v in res["pairs_by_turn_number"].values())
    assert res["headline_heldout_matched"]["pairs"] == pairs


def test_a_conversation_does_not_enter_its_own_fit():
    acts, _, _, conv, _, ref, rng = planted(3, 1.0)
    _, comps, _ = hc.heldout_projections(acts, conv, ref)
    changed = acts.copy()
    changed[conv == 4] += 50.0 * rng.standard_normal((int((conv == 4).sum()), acts.shape[1]))
    _, comps2, _ = hc.heldout_projections(changed, conv, ref)
    assert np.allclose(comps[4], comps2[4]) and not np.allclose(comps[5], comps2[5])


def test_matched_auc_is_the_sum_over_turn_numbers():
    acts, others, drift, conv, turn, ref, rng = planted(4, 0.5)
    held, _, _ = hc.heldout_projections(acts, conv, ref)
    cells = hc.cells_by(np.column_stack([held, others]), drift, conv, 30, turn)
    num = den = 0.0
    for t in np.unique(turn):
        p, q = held[(turn == t) & drift], held[(turn == t) & ~drift]
        num += ((p[:, None] < q[None, :]) + 0.5 * (p[:, None] == q[None, :])).sum()
        den += len(p) * len(q)
    assert np.isclose(hc.auc_cells(cells)[0], num / den)
