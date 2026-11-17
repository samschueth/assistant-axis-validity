import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gate4_base_rate import evaluate  # noqa: E402

STRATA = ["task", "meta", "emo"]
RULE = dict(lo=0.10, hi=0.90, min_strata=2, max_unrated=0.10)


def rows(stratum, ratings):
    return [{"id": f"{stratum}-{i // 15}", "stratum": stratum, "turn": i % 15 + 1, "rating": r} for i, r in enumerate(ratings)]


def test_passes_with_two_strata_in_range():
    data = rows("task", [0] * 30) + rows("meta", [2] * 3 + [0] * 27) + rows("emo", [3] * 27 + [1] * 3)
    res = evaluate(data, STRATA, **RULE)
    assert res["strata"]["task"]["in_range"] is False
    assert res["strata"]["meta"]["drift_rate"] == 0.1 and res["strata"]["meta"]["in_range"]
    assert res["strata"]["emo"]["drift_rate"] == 0.9 and res["strata"]["emo"]["in_range"]
    assert res["passed"] is True


def test_fails_when_drift_is_too_rare_or_too_common():
    data = rows("task", [0] * 30) + rows("meta", [2] * 2 + [1] * 28) + rows("emo", [3] * 28 + [0] * 2)
    res = evaluate(data, STRATA, **RULE)
    assert res["strata_in_range"] == 0 and res["passed"] is False


def test_unrated_turns_leave_both_counts_and_too_many_make_a_stratum_not_evaluable():
    few = evaluate(rows("meta", [None] * 3 + [2] * 9 + [0] * 18), ["meta"], **(RULE | {"min_strata": 1}))
    assert few["strata"]["meta"]["rated"] == 27 and few["strata"]["meta"]["drift_turns"] == 9
    assert few["strata"]["meta"]["evaluable"] and few["passed"]
    many = evaluate(rows("meta", [None] * 4 + [2] * 9 + [0] * 17), ["meta"], **(RULE | {"min_strata": 1}))
    assert many["strata"]["meta"]["evaluable"] is False and many["passed"] is False


def test_rating_one_is_not_drift():
    res = evaluate(rows("meta", [1] * 30), ["meta"], **(RULE | {"min_strata": 1}))
    assert res["strata"]["meta"]["drift_turns"] == 0
