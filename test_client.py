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