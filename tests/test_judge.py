import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from judge import check_same_text, cost_usd, output_names, parse_rating, rate_turn, rubric_text, turn_request  # noqa: E402

TURNS = [{"user": f"u{i}", "reply": f"a{i}"} for i in range(1, 4)]


class FakeClient:
    def __init__(self, text, stop_reason="end_turn"):
        self.text, self.stop_reason, self.seen = text, stop_reason, None

    def complete(self, **kwargs):
        self.seen = kwargs
        return SimpleNamespace(text=self.text, content=[], stop_reason=self.stop_reason, model="m")


def test_rubric_is_read_from_the_preregistration():
    text = rubric_text((Path(__file__).resolve().parent.parent / "PREREGISTRATION.md").read_text())
    assert text.startswith("### 7.1 What is rated")
    assert "### 7.5 Edge cases" in text and "If torn between two adjacent ratings" in text
    assert "Judge validation" not in text


def test_request_shows_the_conversation_up_to_the_turn_and_nothing_else():
    (message,) = turn_request(TURNS, 2)
    text = "".join(b["text"] for b in message["content"])
    assert "u1" in text and "a2" in text and "u3" not in text and "a3" not in text
    assert message["content"][-1]["text"].startswith("Rate the last ASSISTANT turn")


def test_request_for_the_next_turn_repeats_the_earlier_blocks_exactly():
    two = turn_request(TURNS, 2)[0]["content"]
    three = turn_request(TURNS, 3)[0]["content"]
    assert [b["text"] for b in three[:2]] == [b["text"] for b in two[:2]]


def test_parse_accepts_only_valid_ratings():
    assert parse_rating('{"rating": 2, "signs": ["C", "A", "C"], "note": "x"}') == {"rating": 2, "signs": ["A", "C"], "note": "x"}
    assert parse_rating('{"rating": 4, "signs": [], "note": ""}') is None
    assert parse_rating('{"rating": true, "signs": [], "note": ""}') is None
    assert parse_rating("not json") is None


def test_declined_unparseable_and_empty_turns_are_unrated():
    kw = dict(model="m", effort="high", system="s", turns=TURNS, k=1)
    assert rate_turn(FakeClient("", "refusal"), **kw)["rating"] is None
    assert rate_turn(FakeClient("oops"), **kw)["status"].startswith("unparseable")
    empty = [{"user": "u", "reply": ""}]
    client = FakeClient("{}")
    assert rate_turn(client, model="m", effort="high", system="s", turns=empty, k=1)["status"] == "empty reply"
    assert client.seen is None  # no call is made for an empty reply
    ok = rate_turn(FakeClient(json.dumps({"rating": 0, "signs": [], "note": "plain"})), **kw)
    assert ok["rating"] == 0 and ok["status"] == "rated"


def test_one_retry_for_invalid_output_and_none_after_a_refusal():
    class Sequence:
        def __init__(self, outs):
            self.outs, self.n = outs, 0

        def complete(self, **kwargs):
            self.n += 1
            text, stop = self.outs[self.n - 1]
            return SimpleNamespace(text=text, content=[], stop_reason=stop, model="m")

    kw = dict(model="m", effort="high", system="s", turns=TURNS, k=1)
    good = json.dumps({"rating": 1, "signs": [], "note": ""})
    c = Sequence([("", "max_tokens"), (good, "end_turn")])
    assert rate_turn(c, **kw)["rating"] == 1 and c.n == 2
    c = Sequence([("", "max_tokens"), ("", "max_tokens")])
    assert rate_turn(c, **kw)["rating"] is None and c.n == 2
    c = Sequence([("", "refusal"), (good, "end_turn")])
    assert rate_turn(c, **kw)["status"] == "judge declined" and c.n == 1


def test_pending_skips_rated_turns_and_keeps_to_the_sample():
    from judge import pending

    convos = [{"id": "a", "turns": [{}] * 3}, {"id": "b", "turns": [{}] * 2}, {"id": "c", "turns": [{}]}]
    all_jobs = pending(convos, done={("a", 1), ("c", 1)}, only=None)
    assert [(c["id"], ks) for c, ks in all_jobs] == [("a", [2, 3]), ("b", [1, 2])]
    sampled = pending(convos, done={("a", 3)}, only={("a", 2), ("a", 3), ("b", 2)})
    assert [(c["id"], ks) for c, ks in sampled] == [("a", [2]), ("b", [2])]
    assert pending(convos, done=set(), only=set()) == []


def test_each_version_of_the_rubric_has_its_own_ratings_file_and_log():
    assert output_names(1, True, None, None) == ("ratings.jsonl", "judge_log.sample.json")
    assert output_names(1, False, None, None) == ("ratings.jsonl", "judge_log.json")
    assert output_names(2, True, 1, 40) == ("ratings.rubric2.jsonl", "judge_log.rubric2.sample.items1-40.json")
    assert output_names(2, True, 41, None) == ("ratings.rubric2.jsonl", "judge_log.rubric2.sample.items41-end.json")
    assert output_names(2, False, None, None) == ("ratings.rubric2.jsonl", "judge_log.rubric2.json")


def test_a_run_stops_if_the_ratings_file_was_started_under_another_text(tmp_path):
    now = {"rubric_sha256": "new", "instructions_sha256": "i"}
    check_same_text(tmp_path, 1, "ratings.jsonl", now)  # no ratings yet: nothing to protect
    (tmp_path / "ratings.jsonl").write_text("{}\n")
    (tmp_path / "judge_log.sample.json").write_text(json.dumps({"rubric_sha256": "old", "instructions_sha256": "i"}))
    with pytest.raises(SystemExit):
        check_same_text(tmp_path, 1, "ratings.jsonl", now)
    (tmp_path / "ratings.rubric2.jsonl").write_text("{}\n")
    (tmp_path / "judge_log.rubric2.sample.items1-40.json").write_text(json.dumps(now))
    check_same_text(tmp_path, 2, "ratings.rubric2.jsonl", now)  # the version 1 log is not consulted
    with pytest.raises(SystemExit):
        check_same_text(tmp_path, 2, "ratings.rubric2.jsonl", now | {"instructions_sha256": "changed"})


def test_cost_of_a_run():
    usage = {"m": {"calls": 2, "input": 1_000_000, "cache_write": 2_000_000, "cache_read": 10_000_000, "output": 500_000}}
    price = {"input": 4.0, "cache_write": 5.0, "cache_read": 0.2, "output": 20.0}
    assert cost_usd(usage, price) == 4 + 10 + 2 + 10
