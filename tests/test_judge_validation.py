import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from judge_validation import agreement, cross_table, validate  # noqa: E402


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
