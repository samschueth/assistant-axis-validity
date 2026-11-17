"""Thin wrappers around the two APIs used: one for the simulated user, one for the judge.

Everything that calls a paid model goes through `complete`, so tests can pass a fake with the
same method, and so every call's token usage is counted in one place.

Two things are deliberate:
  - No sampling parameters are sent. The current models reject them, so "temperature 0" in the
    pre-registration is not available; effort is the only setting, and it is recorded.
  - No fallback model is configured. If the model declines a request, the caller records the
    refusal. A silent switch to another model would change the study's instruments.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Completion:
    text: str
    content: list  # the response's content blocks, passed back unchanged on the next request
    stop_reason: str
    model: str


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    by_model: dict = field(default_factory=dict)

    def add(self, model: str, *, input: int, cache_write: int, cache_read: int, output: int) -> None:
        """`input` is uncached input tokens; cached ones are counted separately."""
        row = self.by_model.setdefault(model, {"calls": 0, "input": 0, "cache_write": 0, "cache_read": 0, "output": 0})
        counts = {"input": input, "cache_write": cache_write, "cache_read": cache_read, "output": output}
        row["calls"] += 1
        for k, v in counts.items():
            row[k] += v
        self.calls += 1
        self.input_tokens += counts["input"]
        self.cache_write_tokens += counts["cache_write"]
        self.cache_read_tokens += counts["cache_read"]
        self.output_tokens += counts["output"]


class AnthropicClient:
    """Reads its credentials from the environment (ANTHROPIC_API_KEY)."""

    def __init__(self) -> None:
        import anthropic  # imported here so that tests and the analysis never need the package

        self._client = anthropic.Anthropic(max_retries=5)
        self.usage = Usage()

    def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list,
        max_tokens: int,
        effort: str,
        schema: dict | None = None,
    ) -> Completion:
        output_config: dict = {"effort": effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            output_config=output_config,
            # Caches the conversation so far, so each request pays full price only for what the
            # last exchange added. The marker on the system block above keeps its own read point.
            cache_control={"type": "ephemeral"},
        )
        u = response.usage
        self.usage.add(
            response.model,
            input=u.input_tokens or 0,
            cache_write=u.cache_creation_input_tokens or 0,
            cache_read=u.cache_read_input_tokens or 0,
            output=u.output_tokens or 0,
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        return Completion(text=text, content=response.content, stop_reason=response.stop_reason, model=response.model)


class OpenAIClient:
    """Reads its credentials from the environment (OPENAI_API_KEY). Uses the Responses API.

    Takes the same arguments as AnthropicClient.complete. Message content may be a string or a
    list of text blocks; block-level cache markers are dropped, since this API caches repeated
    prefixes on its own.
    """

    def __init__(self) -> None:
        import openai

        self._client = openai.OpenAI(max_retries=5)
        self.usage = Usage()

    def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list,
        max_tokens: int,
        effort: str,
        schema: dict | None = None,
    ) -> Completion:
        items = []
        for m in messages:
            blocks = m["content"] if isinstance(m["content"], list) else [{"text": m["content"]}]
            kind = "output_text" if m["role"] == "assistant" else "input_text"
            items.append({"role": m["role"], "content": [{"type": kind, "text": b["text"]} for b in blocks]})
        kwargs: dict = {}
        if schema is not None:
            kwargs["text"] = {"format": {"type": "json_schema", "name": "result", "schema": schema, "strict": True}}
        response = self._client.responses.create(
            model=model,
            instructions=system,
            input=items,
            max_output_tokens=max_tokens,
            reasoning={"effort": effort},
            **kwargs,
        )
        u = response.usage
        cached = getattr(u.input_tokens_details, "cached_tokens", 0) or 0
        self.usage.add(response.model, input=u.input_tokens - cached, cache_write=0, cache_read=cached, output=u.output_tokens)
        refused = any(
            getattr(part, "type", None) == "refusal"
            for item in response.output
            if getattr(item, "type", None) == "message"
            for part in item.content
        )
        if refused:
            stop = "refusal"
        elif response.status == "incomplete":
            stop = "max_tokens"
        else:
            stop = "end_turn"
        return Completion(text=response.output_text or "", content=[], stop_reason=stop, model=response.model)


def client_for(provider: str):
    return {"anthropic": AnthropicClient, "openai": OpenAIClient}[provider]()
