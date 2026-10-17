import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any

import anthropic

from cache import DiskCache, make_cache_key


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cached: bool = False


RETRYABLE_ERRORS = (
    anthropic.RateLimitError,        # 429: too many requests
    anthropic.InternalServerError,   # 5xx: server trouble
    anthropic.APIConnectionError,    # network problems, and timeouts
)


def backoff_delay(attempt: int, *, base: float = 1.0, cap: float = 30.0) -> float:
    upper_limit = min(cap, base * (2 ** attempt))   # doubles each attempt, never above the cap
    return random.uniform(0, upper_limit)           # a random wait up to that limit


class LLMClient:
    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        cache: DiskCache | None = None,
    ):
        self.model = model
        self.max_retries = max_retries
        self.cache = cache
        self._sleep = time.sleep      # kept on the object so a test can swap in a fake
        self._client = anthropic.Anthropic(
            max_retries=0,            # the SDK's own retries stay off
            timeout=timeout_seconds,
        )

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 300,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> Completion:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            request["temperature"] = temperature
        if top_p is not None:
            request["top_p"] = top_p

        key = make_cache_key(request) if self.cache is not None else None
        if self.cache is not None and key is not None:
            saved = self.cache.get(key)
            if saved is not None:
                return Completion(
                    text=saved["text"],
                    input_tokens=saved["input_tokens"],
                    output_tokens=saved["output_tokens"],
                    latency_ms=0.0,
                    cached=True,
                )

        attempt = 0
        while True:
            start_time = time.perf_counter()
            try:
                response = self._client.messages.create(**request)
            except RETRYABLE_ERRORS:
                if attempt >= self.max_retries:
                    raise                                    # out of retries, so show the real error
                self._sleep(backoff_delay(attempt))          # wait, then try again
                attempt += 1
                continue

            latency_ms = (time.perf_counter() - start_time) * 1000
            completion = Completion(
                text=response.content[0].text,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=latency_ms,
            )
            if self.cache is not None and key is not None:
                self.cache.set(key, {
                    "text": completion.text,
                    "input_tokens": completion.input_tokens,
                    "output_tokens": completion.output_tokens,
                })
            return completion


class AsyncLLMClient:
    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        max_concurrent: int = 5,
        cache: DiskCache | None = None,
    ):
        self.model = model
        self.max_retries = max_retries
        self.cache = cache
        self._sleep = asyncio.sleep
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client = anthropic.AsyncAnthropic(
            max_retries=0,
            timeout=timeout_seconds,
        )

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 300,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> Completion:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            request["temperature"] = temperature
        if top_p is not None:
            request["top_p"] = top_p

        key = make_cache_key(request) if self.cache is not None else None
        if self.cache is not None and key is not None:
            saved = self.cache.get(key)
            if saved is not None:
                return Completion(
                    text=saved["text"],
                    input_tokens=saved["input_tokens"],
                    output_tokens=saved["output_tokens"],
                    latency_ms=0.0,
                    cached=True,
                )

        attempt = 0
        while True:
            start_time = time.perf_counter()
            try:
                async with self._semaphore:
                    response = await self._client.messages.create(**request)
            except RETRYABLE_ERRORS:
                if attempt >= self.max_retries:
                    raise
                await self._sleep(backoff_delay(attempt))
                attempt += 1
                continue

            latency_ms = (time.perf_counter() - start_time) * 1000
            completion = Completion(
                text=response.content[0].text,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=latency_ms,
            )
            if self.cache is not None and key is not None:
                self.cache.set(key, {
                    "text": completion.text,
                    "input_tokens": completion.input_tokens,
                    "output_tokens": completion.output_tokens,
                })
            return completion

    async def complete_many(self, prompts: list[str], **kwargs: Any) -> list[Completion | BaseException]:
        tasks = [self.complete(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)