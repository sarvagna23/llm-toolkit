import time
from dataclasses import dataclass

import anthropic


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMClient:
    def __init__(self, model: str = "claude-sonnet-5-5"):
        self.model = model
        self._client = anthropic.Anthropic()    # reads ANTHROPIC_API_KEY from the environment

    def complete(self, prompt: str, *, max_tokens: int = 300) -> Completion:
        start_time = time.perf_counter()

        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return Completion(
            text=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
        )