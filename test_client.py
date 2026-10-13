import anthropic
import httpx
import pytest

from client import backoff_delay
from types import SimpleNamespace

from client import Completion, LLMClient


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


def test_complete_returns_completion(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")    # so LLMClient() can be created
    client = LLMClient()
    client._client = FakeAnthropic()                       # swap the real SDK for the fake

    result = client.complete("hi", max_tokens=20)

    assert isinstance(result, Completion)
    assert result.text == "Hello from the fake"
    assert result.input_tokens == 11
    assert result.output_tokens == 4
    assert result.latency_ms >= 0

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