"""Needs the model's tokenizer, so it runs only where transformers is installed and the
tokenizer files can be fetched (the GPU instance, or a laptop with both)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from capture import last_reply_span  # noqa: E402

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tok():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    except Exception as e:  # no network, or no access
        pytest.skip(f"tokenizer not available: {e}")


def helpers(tok):
    def render(messages, add_generation_prompt):
        out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt, enable_thinking=False)
        return list(out["input_ids"] if hasattr(out, "keys") else out)

    return render, lambda text: tok(text, add_special_tokens=False)["input_ids"]


REPLIES = ["Sure. Here is the formula:\n\n=SUM(A1:A9)", "i don't know.", "1. First\n2. Second", "Ünïcode — and emoji 🙂 too"]


@pytest.mark.parametrize("n_prior", [0, 1, 3])
@pytest.mark.parametrize("reply", REPLIES)
def test_reply_span_is_the_reply_and_its_context_is_the_generation_prompt(tok, n_prior, reply):
    render, encode = helpers(tok)
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(n_prior):
        messages += [{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": f"answer {i}"}]
    messages += [{"role": "user", "content": "next question"}, {"role": "assistant", "content": reply}]
    span = last_reply_span(render, encode, messages)
    assert tok.decode(span["ids"][span["start"] : span["end"]]) == reply
    assert span["matches_generation"] is True
    # The empty thinking block sits directly before the reply.
    assert tok.decode(span["ids"][span["start"] - 4 : span["start"]]).strip().endswith("</think>")
