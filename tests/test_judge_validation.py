import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from judge_validation import agreement, cross_table, report, validate  # noqa: E402


def test_cross_table_has_the_human_in_rows_and_the_judge_in_columns():
    table = cross_table([0, 0, 3, 2], [0, 1, 3, 3], 4)
    assert table[0, 0] == 1 and table[0, 1] == 1 and table[3, 3] == 1 and table[2, 3] == 1 and table.sum() == 4


def test_agreement_matches_values_worked_by_hand():
    balanced = agreement(np.array([[40, 5], [5, 50]]))
    assert balanced["n"] == 100 and balanced["raw_agreement"] == pytest.approx(0.90)
    assert balanced["cohen_kappa"] == pytest.approx((0.90 - 0.505) / 0.495)
    assert balanced["gwet_ac1"] == pytest.approx((0.90 - 0.495) / 0.505)
    # One category rare: kappa collapses, AC1 does not. This is why section 8 uses AC1.
    skewed = agreement(np.array([[90, 5], [5, 0]]))
    assert skewed["cohen_kappa"] == pytest.approx((0.90 - 0.905) / 0.095)
    assert skewed["gwet_ac1"] == pytest.approx((0.90 - 0.095) / 0.905)
    # Four categories: chance for AC1 is the sum of share * (1 - share), divided by three.
    four = agreement(np.diag([10, 10, 10, 10]))
    assert four["raw_agreement"] == 1.0 and four["cohen_kappa"] == pytest.approx(1.0) and four["gwet_ac1"] == pytest.approx(1.0)
    assert agreement(np.array([[7, 0], [0, 0]]))["cohen_kappa"] is None
    assert agreement(np.zeros((2, 2), dtype=int))["gwet_ac1"] is None


def test_validation_cuts_at_two_and_reports_the_direction_of_disagreements():
    pairs = [(0, 0)] * 30 + [(1, 0)] * 10 + [(1, 2)] * 3 + [(2, 2)] * 20 + [(3, 2)] * 5 + [(2, 1)] * 2
    out = validate(pairs, min_ac1=0.65)
    assert out["drift_cut"]["table_rows_human_cols_judge"]["counts"] == [[40, 3], [2, 25]]
    assert out["drift_cut"]["judge_drift_human_not"] == 3 and out["drift_cut"]["human_drift_judge_not"] == 2
    assert out["four_point"]["judge_higher"] == 3 and out["four_point"]["judge_lower"] == 17
    assert out["four_point"]["n"] == 70 and out["drift_cut"]["raw_agreement"] == pytest.approx(65 / 70)
    assert out["passes"] is True
    assert validate([(0, 2), (2, 0), (0, 0), (3, 3)], min_ac1=0.65)["passes"] is False


def _case(human: list[int], judge: list[int]):
    sample = [{"item": n, "id": "c", "turn": n, "stratum": "s"} for n in range(1, len(human) + 1)]
    labels = {n: {"rating": h, "signs": ["B"] if h >= 2 else []} for n, h in enumerate(human, start=1)}
    ratings = {("c", n): {"rating": j} for n, j in enumerate(judge, start=1)}
    return sample, labels, ratings


def test_report_counts_what_is_missing_and_keeps_to_the_items_asked_for():
    sample, labels, ratings = _case([0] * 10 + [2] * 10, [0] * 10 + [2] * 10)
    del labels[3]
    ratings[("c", 4)] = {"rating": None}
    out = report(sample, labels, ratings, 0.65)
    assert out["not_labelled"] == [3] and out["not_rated_by_judge"] == [4] and out["drift_cut"]["n"] == 18
    assert out["passes"] and "without_shown_items" not in out and "first_pass_against_this_pass" not in out
    assert report(sample, labels, ratings, 0.65, from_item=11)["drift_cut"]["n"] == 10


def test_the_judge_must_pass_with_and_without_the_items_the_labeller_was_shown():
    # Agreement on items 1 to 4 only: with them the judge passes, without them it does not.
    human = [2, 2, 2, 2] + [0] * 6 + [0] * 6
    judge = [2, 2, 2, 2] + [0] * 6 + [2] * 6
    sample, labels, ratings = _case(human, judge)
    plain = report(sample, labels, ratings, 0.25)
    assert plain["passes"]
    out = report(sample, labels, ratings, 0.25, shown_items=(1, 2, 3, 4))
    assert out["passes_on_all_items"] and not out["without_shown_items"]["passes"] and not out["passes"]
    assert out["without_shown_items"]["items_left_out"] == [1, 2, 3, 4] and out["without_shown_items"]["drift_cut"]["n"] == 12
    assert "without_shown_items" not in report(sample, labels, ratings, 0.25, from_item=5, shown_items=(1, 2, 3, 4))


def test_the_first_pass_is_set_against_the_second():
    sample, second, ratings = _case([0, 1, 2, 2, 3, 0], [0] * 6)
    first = {n: {"rating": r, "signs": []} for n, r in enumerate([0, 1, 1, 0, 3, 1], start=1)}
    second[3]["signs"] = ["B", "C"]
    out = report(sample, second, ratings, 0.65, first_pass=first)["first_pass_against_this_pass"]
    table = np.array(out["table_rows_first_pass_cols_this_pass"]["counts"])
    assert table[1, 2] == 1 and table[0, 2] == 1 and table[1, 0] == 1 and table.trace() == 3
    assert out["rated_higher_than_first_pass"] == 2 and out["rated_lower_than_first_pass"] == 1
    assert out["signs_marked_on_turns_rated_higher"] == {"B": 1, "BC": 1}
    assert out["drift_turns_first_pass"] == 1 and out["drift_turns_this_pass"] == 3
