"""One thin wrapper around the API used for the simulated user and the judge.

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

    def add(self, model: str, u) -> None:
        row = self.by_model.setdefault(model, {"calls": 0, "input": 0, "cache_write": 0, "cache_read": 0, "output": 0})
        counts = {
            "input": u.input_tokens or 0,
            "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
            "output": u.output_tokens or 0,
        }
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
        )
        self.usage.add(response.model, response.usage)
        text = "".join(b.text for b in response.content if b.type == "text")
        return Completion(text=text, content=response.content, stop_reason=response.stop_reason, model=response.model)
