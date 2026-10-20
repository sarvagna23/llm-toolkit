from types import SimpleNamespace
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from cache import DiskCache
from client import LLMClient
from structured import StructuredOutputError


class Ticket(BaseModel):
    category: Literal["billing", "bug", "other"]
    urgent: bool
    summary: str


class JustCategory(BaseModel):
    category: str


class Rating(BaseModel):
    score: int = Field(ge=1, le=5)


GOOD = '{"category": "billing", "urgent": true, "summary": "Charged twice"}'


class ScriptedMessages:
    def __init__(self, text, stop_reason="end_turn"):
        self.text = text
        self.stop_reason = stop_reason
        self.calls = 0
        self.last_kwargs = None

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(text=self.text)],
            usage=SimpleNamespace(input_tokens=20, output_tokens=10),
            stop_reason=self.stop_reason,
        )


def make_client(monkeypatch, text, stop_reason="end_turn"):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = LLMClient()
    messages = ScriptedMessages(text, stop_reason)
    client._client = SimpleNamespace(messages=messages)
    return client, messages


def test_structured_returns_a_validated_object(monkeypatch):
    client, messages = make_client(monkeypatch, GOOD)
    result = client.complete_structured("I was charged twice", Ticket)
    assert isinstance(result.data, Ticket)
    assert result.data.category == "billing"
    assert result.data.urgent is True
    assert result.input_tokens == 20


def test_structured_sends_the_schema_in_output_config(monkeypatch):
    client, messages = make_client(monkeypatch, GOOD)
    client.complete_structured("hi", Ticket)
    sent = messages.last_kwargs["output_config"]["format"]
    assert sent["type"] == "json_schema"
    assert set(sent["schema"]["properties"]) == {"category", "urgent", "summary"}
    assert sent["schema"]["additionalProperties"] is False


def test_structured_rejects_a_value_outside_the_literal(monkeypatch):
    client, messages = make_client(
        monkeypatch, '{"category": "refund", "urgent": true, "summary": "x"}'
    )
    with pytest.raises(StructuredOutputError):
        client.complete_structured("hi", Ticket)


def test_structured_rejects_text_that_is_not_json(monkeypatch):
    client, messages = make_client(monkeypatch, "Here is your ticket: billing")
    with pytest.raises(StructuredOutputError):
        client.complete_structured("hi", Ticket)


def test_structured_reports_a_cut_off_reply(monkeypatch):
    client, messages = make_client(
        monkeypatch, '{"category": "billing", "urg', stop_reason="max_tokens"
    )
    with pytest.raises(StructuredOutputError) as info:
        client.complete_structured("hi", Ticket)
    assert info.value.stop_reason == "max_tokens"


def test_structured_reports_a_refusal(monkeypatch):
    client, messages = make_client(monkeypatch, "I can't help with that.", stop_reason="refusal")
    with pytest.raises(StructuredOutputError) as info:
        client.complete_structured("hi", Ticket)
    assert info.value.stop_reason == "refusal"


def test_constraints_the_api_cannot_enforce_are_checked_locally(monkeypatch):
    client, messages = make_client(monkeypatch, '{"score": 7}')
    with pytest.raises(StructuredOutputError):
        client.complete_structured("rate it", Rating)


def test_structured_results_come_from_the_cache_on_repeat(monkeypatch, tmp_path):
    client, messages = make_client(monkeypatch, GOOD)
    client.cache = DiskCache(str(tmp_path))

    first = client.complete_structured("hi", Ticket)
    second = client.complete_structured("hi", Ticket)

    assert messages.calls == 1
    assert first.cached is False
    assert second.cached is True
    assert second.data == first.data


def test_a_different_response_model_is_a_different_cache_entry(monkeypatch, tmp_path):
    client, messages = make_client(monkeypatch, GOOD)
    client.cache = DiskCache(str(tmp_path))

    client.complete_structured("hi", Ticket)
    client.complete_structured("hi", JustCategory)

    assert messages.calls == 2


def test_cut_off_replies_are_not_cached(monkeypatch, tmp_path):
    client, messages = make_client(monkeypatch, '{"category": "bil', stop_reason="max_tokens")
    client.cache = DiskCache(str(tmp_path))

    for _ in range(2):
        with pytest.raises(StructuredOutputError):
            client.complete_structured("hi", Ticket)

    assert messages.calls == 2


def test_plain_complete_does_not_cache_cut_off_replies(monkeypatch, tmp_path):
    client, messages = make_client(monkeypatch, "half a sent", stop_reason="max_tokens")
    client.cache = DiskCache(str(tmp_path))

    client.complete("hi")
    client.complete("hi")

    assert messages.calls == 2