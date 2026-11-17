"""Where a reply's tokens sit in the rendered conversation, for activation capture.

Each assistant turn is captured from a forward pass over the conversation cut off at that turn,
so the reply being measured is always the last message. This matters for this model's chat
template: with thinking off it writes an empty thinking block before the last assistant message
only. A single pass over the whole conversation would therefore show every earlier reply in a
context that differs from the one it was generated in, and from the one gate 3 validated.
"""

from __future__ import annotations

from typing import Callable

Render = Callable[[list, bool], list[int]]  # (messages, add_generation_prompt) -> token ids


def last_reply_span(render: Render, encode: Callable[[str], list[int]], messages: list) -> dict:
    """Token ids of the conversation ending in an assistant reply, and that reply's [start, end).

    The span covers the reply's own tokens: the role header and the empty thinking block before
    it and the end-of-turn marker after it are left out. `matches_generation` says whether the
    tokens before the reply are exactly the prompt the reply was generated from.
    """
    if messages[-1]["role"] != "assistant":
        raise ValueError("the last message must be the assistant reply to capture")
    full = render(messages, False)
    prompt = render(messages[:-1], True)
    content = encode(messages[-1]["content"])
    a = len(render(messages[:-1], False))
    block = full[a:]
    start = next(
        (h for h in range(len(block) - len(content) + 1) if block[h : h + len(content)] == content),
        None,
    )
    if start is None:
        raise ValueError("could not locate the reply in the rendered conversation")
    begin = a + start
    return {
        "ids": full,
        "start": begin,
        "end": begin + len(content),
        "matches_generation": full[:begin] == prompt,
    }
