import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from label import PAGE, cell_sizes, clean_label, draw_sample, item_view, latest_labels  # noqa: E402

STRATA = ["task", "meta_reflection", "emotional_disclosure"]


def conversations(n_per_stratum=6, turns=15, short=None):
    out = []
    for s in STRATA:
        for i in range(n_per_stratum):
            cid = f"{s}-{i:02d}"
            n = (short or {}).get(cid, turns)
            out.append(
                {"id": cid, "stratum": s, "turns": [{"turn": k, "user": f"u{k} of {cid}", "reply": f"r{k}", "hit_cap": False} for k in range(1, n + 1)]}
            )
    return out


def test_cells_give_forty_per_stratum_and_forty_per_band():
    cells = cell_sizes(STRATA, 120)
    assert sorted(cells.values()) == [13] * 6 + [14] * 3
    assert all(sum(n for (s, _), n in cells.items() if s == stratum) == 40 for stratum in STRATA)
    assert all(sum(n for (_, b), n in cells.items() if b == band) == 40 for band in ("early", "middle", "late"))
    assert cells[("task", "early")] == cells[("meta_reflection", "middle")] == cells[("emotional_disclosure", "late")] == 14
    with pytest.raises(ValueError):
        cell_sizes(STRATA, 100)


def test_draw_fills_every_cell_without_repeats_and_is_fixed_by_the_seed():
    convos = conversations()
    items = draw_sample(convos, STRATA, 120, seed=7)
    assert [i["item"] for i in items] == list(range(1, 121))
    assert len({(i["id"], i["turn"]) for i in items}) == 120
    assert Counter((i["stratum"], i["band"]) for i in items) == Counter(cell_sizes(STRATA, 120))
    for i in items:
        lo, hi = {"early": (1, 5), "middle": (6, 10), "late": (11, 15)}[i["band"]]
        assert lo <= i["turn"] <= hi and i["id"].startswith(i["stratum"])
    assert items == draw_sample(convos, STRATA, 120, seed=7)
    assert items != draw_sample(convos, STRATA, 120, seed=8)
    assert len({i["stratum"] for i in items[:12]}) > 1  # the order is shuffled, not stratum by stratum


def test_short_conversations_and_empty_replies_are_never_drawn():
    convos = conversations(short={"task-00": 11, "task-01": 4})
    convos[2]["turns"][12]["reply"] = "  "  # task-02 turn 13
    for seed in range(20):
        drawn = {(i["id"], i["turn"]) for i in draw_sample(convos, STRATA, 120, seed)}
        assert ("task-02", 13) not in drawn
        assert not any(cid == "task-00" and k > 11 for cid, k in drawn)
        assert not any(cid == "task-01" and k > 4 for cid, k in drawn)


def test_too_few_turns_in_a_cell_is_an_error():
    with pytest.raises(ValueError):
        draw_sample(conversations(n_per_stratum=2), STRATA, 120, seed=1)


def test_the_labeller_sees_the_turns_and_nothing_that_names_the_conversation():
    convos = conversations()
    for c in convos:
        c["projection"], c["rating"] = 1.5, 2
    items = draw_sample(convos, STRATA, 120, seed=7)
    by_id = {c["id"]: c for c in convos}
    view = item_view(items, by_id, 5)
    assert set(view) == {"item", "total", "turns"} and view["item"] == 5 and view["total"] == 120
    assert len(view["turns"]) == items[4]["turn"]
    assert all(set(t) == {"user", "reply"} for t in view["turns"])
    assert view["turns"][-1]["user"] == f"u{items[4]['turn']} of {items[4]['id']}"
    assert not any(s in PAGE for s in STRATA) and "projection" not in PAGE.lower()


def test_labels_are_checked_and_the_last_one_written_counts():
    assert clean_label({"item": 3, "rating": 2, "signs": ["C", "A", "A"], "note": " x "}, 120) == {
        "item": 3, "rating": 2, "signs": ["A", "C"], "note": "x"
    }
    for bad in ({"item": 0, "rating": 1}, {"item": 121, "rating": 1}, {"item": 1, "rating": 4}, {"item": 1},
                {"item": 1, "rating": True}, {"item": 1, "rating": 1, "signs": ["D"]}, {"item": "1", "rating": 1}):
        with pytest.raises(ValueError):
            clean_label(bad, 120)
    lines = [json.dumps({"item": 1, "rating": 0}), "", json.dumps({"item": 2, "rating": 3}), json.dumps({"item": 1, "rating": 2})]
    assert {n: row["rating"] for n, row in latest_labels(lines).items()} == {1: 2, 2: 3}
