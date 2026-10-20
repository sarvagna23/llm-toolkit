import asyncio
from types import SimpleNamespace

from pydantic import BaseModel

from cache import DiskCache
from client import AsyncLLMClient
from structured import StructuredCompletion, StructuredOutputError


class Ticket(BaseModel):
    category: str
    urgent: bool


GOOD = '{"category": "billing", "urgent": true}'


class PromptDrivenMessages:
    """Replies with bad text if the prompt contains the word 'bad', and valid JSON otherwise."""

    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        prompt = kwargs["messages"][0]["content"]
        text = "not json" if "bad" in prompt else GOOD
        return SimpleNamespace(
            content=[SimpleNamespace(text=text)],
            usage=SimpleNamespace(input_tokens=5, output_tokens=3),
            stop_reason="end_turn",
        )


class CutOffMessages:
    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            content=[SimpleNamespace(text="half a sent")],
            usage=SimpleNamespace(input_tokens=5, output_tokens=3),
            stop_reason="max_tokens",
        )


def make_async_client(monkeypatch, messages, **kwargs):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = AsyncLLMClient(**kwargs)
    client._client = SimpleNamespace(messages=messages)
    return client


def test_async_structured_returns_a_validated_object(monkeypatch):
    client = make_async_client(monkeypatch, PromptDrivenMessages())

    result = asyncio.run(client.complete_structured("hi", Ticket))

    assert isinstance(result, StructuredCompletion)
    assert result.data == Ticket(category="billing", urgent=True)


def test_async_structured_many_keeps_order_and_isolates_failures(monkeypatch):
    client = make_async_client(monkeypatch, PromptDrivenMessages())

    results = asyncio.run(
        client.complete_structured_many(["ok 1", "bad one", "ok 2"], Ticket)
    )

    assert isinstance(results[0], StructuredCompletion)
    assert isinstance(results[1], StructuredOutputError)
    assert isinstance(results[2], StructuredCompletion)


def test_async_plain_complete_does_not_cache_cut_off_replies(monkeypatch, tmp_path):
    messages = CutOffMessages()
    client = make_async_client(monkeypatch, messages, cache=DiskCache(str(tmp_path)))

    async def run_twice():
        await client.complete("hi")
        await client.complete("hi")

    asyncio.run(run_twice())

    assert messages.calls == 2


def test_async_structured_results_come_from_the_cache_on_repeat(monkeypatch, tmp_path):
    messages = PromptDrivenMessages()
    client = make_async_client(monkeypatch, messages, cache=DiskCache(str(tmp_path)))

    async def run_twice():
        first = await client.complete_structured("hi", Ticket)
        second = await client.complete_structured("hi", Ticket)
        return first, second

    first, second = asyncio.run(run_twice())

    assert messages.calls == 1
    assert first.cached is False
    assert second.cached is True
    assert second.data == first.data