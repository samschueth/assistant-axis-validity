import json
import sys
import tomllib
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from merge_shards import merge  # noqa: E402


def load(name):
    with open(ROOT / "configs" / name, "rb") as f:
        return tomllib.load(f)


BRIEFS = load("main_briefs.toml")["briefs"]
CFG = load("study.toml")


def test_counts_match_the_configured_sample():
    assert dict(Counter(b["stratum"] for b in BRIEFS)) == CFG["sample"]["conversations"]


def test_ids_are_unique_and_none_is_a_pilot_brief():
    ids = [b["id"] for b in BRIEFS]
    assert len(set(ids)) == len(ids)
    assert not any(i.startswith("pilot") for i in ids)
    pilot_texts = {b["brief"] for b in load("pilot_briefs.toml")["briefs"]}
    assert not any(b["brief"] in pilot_texts for b in BRIEFS)


def test_every_brief_follows_the_recipe():
    for b in BRIEFS:
        text = b["brief"]
        for part in ("Who:", "Opening:", "Course:", "Style:", "messages 1-5", "Messages 6-10", "Messages 11-15"):
            assert part in text, (b["id"], part)


def test_two_shards_split_every_stratum_evenly():
    for shard in (0, 1):
        got = Counter(b["stratum"] for i, b in enumerate(BRIEFS) if i % 2 == shard)
        assert got == {k: v // 2 for k, v in CFG["sample"]["conversations"].items()}


def test_merge_orders_by_brief_and_reports_missing():
    lines = [json.dumps({"id": i}) for i in ("c", "a")]
    merged, missing = merge(["a", "b", "c"], lines)
    assert [json.loads(x)["id"] for x in merged] == ["a", "c"]
    assert missing == ["b"]
    with pytest.raises(ValueError):
        merge(["a"], [json.dumps({"id": "a"})] * 2)
    with pytest.raises(ValueError):
        merge(["a"], [json.dumps({"id": "z"})])
