import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from length_probe import length_table, probe_items  # noqa: E402


def convo(cid, stratum, n):
    return {
        "id": cid,
        "stratum": stratum,
        "turns": [{"user": f"u{i}", "reply": f"r{i}", "new_tokens": 10 * i, "hit_cap": i % 2 == 0} for i in range(1, n + 1)],
    }


def test_context_is_the_pilot_conversation_up_to_the_user_message():
    items = probe_items([convo("a", "task", 5)], [1, 3])
    first, third = items
    assert first["messages"] == [{"role": "user", "content": "u1"}]
    assert [m["role"] for m in third["messages"]] == ["user", "assistant", "user", "assistant", "user"]
    assert [m["content"] for m in third["messages"]] == ["u1", "r1", "u2", "r2", "u3"]
    assert third["pilot_new_tokens"] == 30
    assert all(m["role"] != "system" for i in items for m in i["messages"])


def test_order_interleaves_conversations_and_skips_missing_turns():
    items = probe_items([convo("a", "task", 5), convo("b", "meta_reflection", 2)], [1, 4])
    assert [(i["id"], i["turn"]) for i in items] == [("a", 1), ("b", 1), ("a", 4)]


def test_length_table_shares():
    rows = [{"turn": 1, "new_tokens": n} for n in (100, 400, 800, 2048)]
    t = length_table(rows, "turn")["1"]
    assert t["n"] == 4 and t["lengths"] == [100, 400, 800, 2048]
    assert t["share_reaching"]["384"] == 0.75
    assert t["share_reaching"]["1024"] == 0.25
