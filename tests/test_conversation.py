import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conversation import SimulatedUser, UserModelStopped, run_conversation  # noqa: E402


class FakeClient:
    def __init__(self, refuse_at=None):
        self.calls, self.refuse_at = [], refuse_at

    def complete(self, *, model, system, messages, max_tokens, effort, schema=None):
        self.calls.append({"system": system, "messages": [dict(m) for m in messages]})
        n = len(self.calls)
        if n == self.refuse_at:
            return SimpleNamespace(text="", content=[], stop_reason="refusal", model=model)
        return SimpleNamespace(text=f" person message {n} ", content=[{"type": "text", "text": f"person message {n}"}], stop_reason="end_turn", model=model)


def assistant(messages):
    return {"reply": f"reply to: {messages[-1]['content']}", "reply_tokens": 5}


def run(client, n=3):
    user = SimulatedUser(client, model="m", effort="low", brief="A brief.", n_turns=n)
    return run_conversation(system_prompt="SYS", n_turns=n, next_user=user.next_message, next_assistant=assistant)


def test_roles_alternate_and_turns_are_recorded():
    out = run(FakeClient())
    assert [m["role"] for m in out["messages"]] == ["system", "user", "assistant"] * 1 + ["user", "assistant"] * 2
    assert out["complete"] and out["stopped"] is None
    assert [t["turn"] for t in out["turns"]] == [1, 2, 3]
    assert out["turns"][1]["user"] == "person message 2"
    assert out["turns"][1]["reply"] == "reply to: person message 2"


def test_user_model_sees_roles_swapped_and_never_the_system_prompt():
    client = FakeClient()
    run(client)
    last = client.calls[-1]
    assert "A brief." in last["system"] and "SYS" not in last["system"]
    assert [m["role"] for m in last["messages"]] == ["user", "assistant", "user", "assistant", "user"]
    assert last["messages"][0]["content"].startswith("[The conversation has not started")
    assert last["messages"][2]["content"].startswith("reply to: person message 1")
    assert "message 3 of 3" in last["messages"][4]["content"]
    assert all("SYS" not in str(m["content"]) for m in last["messages"])


def test_refusal_ends_the_conversation_and_marks_it_incomplete():
    out = run(FakeClient(refuse_at=2))
    assert len(out["turns"]) == 1
    assert out["complete"] is False
    assert "refusal" in out["stopped"]
    assert out["messages"][-1]["role"] == "assistant"


def test_cut_off_user_message_is_retried_once():
    class CutOffFirst(FakeClient):
        def complete(self, **kwargs):
            out = super().complete(**kwargs)
            if len(self.calls) == 1:
                out.stop_reason = "max_tokens"
            return out

    client = CutOffFirst()
    out = run(client, n=1)
    assert out["complete"] and len(client.calls) == 2
    assert out["turns"][0]["user"] == "person message 2"


def test_no_system_message_when_the_system_prompt_is_empty():
    user = SimulatedUser(FakeClient(), model="m", effort="low", brief="A brief.", n_turns=2)
    out = run_conversation(system_prompt="", n_turns=2, next_user=user.next_message, next_assistant=assistant)
    assert [m["role"] for m in out["messages"]] == ["user", "assistant", "user", "assistant"]
