import asyncio
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from cache import DiskCache
from client import AsyncLLMClient, Completion, LLMClient, backoff_delay


# ---------- fakes ----------

class FakeMessages:
    def create(self, **kwargs):
        # shaped like the real response: .content[0].text and .usage.input_tokens
        return SimpleNamespace(
            content=[SimpleNamespace(text="Hello from the fake")],
            usage=SimpleNamespace(input_tokens=11, output_tokens=4),
        )


class FakeAnthropic:
    def __init__(self):
        self.messages = FakeMessages()


def make_status_error(error_class, status_code):
    request = httpx.Request("POST", "https://api.example.com/v1/messages")
    response = httpx.Response(status_code, request=request)
    return error_class("fake error", response=response, body=None)


class FlakyMessages:
    def __init__(self, errors):
        self.errors = list(errors)    # one error is raised per call, in order
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return SimpleNamespace(
            content=[SimpleNamespace(text="ok")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )


def make_client(monkeypatch, errors, max_retries=3):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = LLMClient(max_retries=max_retries)
    messages = FlakyMessages(errors)
    client._client = SimpleNamespace(messages=messages)
    sleeps = []
    client._sleep = sleeps.append     # records each wait and never really sleeps
    return client, messages, sleeps


class SlowAsyncMessages:
    def __init__(self):
        self.active = 0
        self.max_active = 0

    async def create(self, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return SimpleNamespace(
            content=[SimpleNamespace(text="ok")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )


class CountingAsyncMessages:
    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            content=[SimpleNamespace(text="ok")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )


class RecordingMessages:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(text="ok")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )


# ---------- basic behavior ----------

def test_complete_returns_completion(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = LLMClient()
    client._client = FakeAnthropic()                       # swap the real SDK for the fake

    result = client.complete("hi", max_tokens=20)

    assert isinstance(result, Completion)
    assert result.text == "Hello from the fake"
    assert result.input_tokens == 11
    assert result.output_tokens == 4
    assert result.latency_ms >= 0


# ---------- retries ----------

def test_retries_then_succeeds(monkeypatch):
    errors = [make_status_error(anthropic.RateLimitError, 429) for _ in range(2)]
    client, messages, sleeps = make_client(monkeypatch, errors)

    result = client.complete("hi")

    assert result.text == "ok"
    assert messages.calls == 3                  # two failures, then success
    assert len(sleeps) == 2
    assert sleeps[0] <= 1 and sleeps[1] <= 2    # the limits for attempts 0 and 1


def test_gives_up_after_max_retries(monkeypatch):
    errors = [make_status_error(anthropic.RateLimitError, 429) for _ in range(4)]
    client, messages, sleeps = make_client(monkeypatch, errors, max_retries=3)

    with pytest.raises(anthropic.RateLimitError):
        client.complete("hi")

    assert messages.calls == 4                  # first try plus 3 retries
    assert len(sleeps) == 3


def test_does_not_retry_auth_error(monkeypatch):
    errors = [make_status_error(anthropic.AuthenticationError, 401)]
    client, messages, sleeps = make_client(monkeypatch, errors)

    with pytest.raises(anthropic.AuthenticationError):
        client.complete("hi")

    assert messages.calls == 1                  # no retry for a permanent error
    assert sleeps == []


def test_retries_connection_error(monkeypatch):
    request = httpx.Request("POST", "https://api.example.com/v1/messages")
    errors = [anthropic.APIConnectionError(request=request)]
    client, messages, sleeps = make_client(monkeypatch, errors)

    result = client.complete("hi")

    assert result.text == "ok"
    assert messages.calls == 2


def test_backoff_delay_stays_in_bounds():
    for attempt in range(10):
        limit = min(30.0, 2 ** attempt)
        for _ in range(200):
            assert 0 <= backoff_delay(attempt) <= limit


# ---------- async client ----------

def test_async_client_respects_max_concurrent(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = AsyncLLMClient(max_concurrent=3)
    messages = SlowAsyncMessages()
    client._client = SimpleNamespace(messages=messages)

    results = asyncio.run(client.complete_many([f"p{i}" for i in range(10)]))

    assert len(results) == 10
    assert all(isinstance(result, Completion) and result.text == "ok" for result in results)
    assert messages.max_active == 3          # never more than 3 at once


# ---------- cache ----------

def test_cache_avoids_second_api_call(monkeypatch, tmp_path):
    client, messages, sleeps = make_client(monkeypatch, errors=[])
    client.cache = DiskCache(str(tmp_path))

    first = client.complete("hi")
    second = client.complete("hi")

    assert messages.calls == 1
    assert first.cached is False
    assert second.cached is True
    assert second.text == first.text
    assert second.input_tokens == first.input_tokens


def test_cache_misses_when_settings_differ(monkeypatch, tmp_path):
    client, messages, sleeps = make_client(monkeypatch, errors=[])
    client.cache = DiskCache(str(tmp_path))

    client.complete("hi", max_tokens=10)
    client.complete("hi", max_tokens=20)
    client.complete("hi", max_tokens=10)    # repeat of the first call

    assert messages.calls == 2


def test_async_cache_avoids_second_api_call(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = AsyncLLMClient(cache=DiskCache(str(tmp_path)))
    messages = CountingAsyncMessages()
    client._client = SimpleNamespace(messages=messages)

    async def run_twice():
        first = await client.complete("hi")
        second = await client.complete("hi")
        return first, second

    first, second = asyncio.run(run_twice())

    assert messages.calls == 1
    assert first.cached is False
    assert second.cached is True
    assert second.text == first.text


# ---------- sampling settings ----------

def test_temperature_and_top_p_are_sent_only_when_given(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = LLMClient()
    messages = RecordingMessages()
    client._client = SimpleNamespace(messages=messages)

    client.complete("hi")
    assert "temperature" not in messages.last_kwargs
    assert "top_p" not in messages.last_kwargs

    client.complete("hi", temperature=0)
    assert messages.last_kwargs["temperature"] == 0

    client.complete("hi", top_p=0.9)
    assert messages.last_kwargs["top_p"] == 0.9