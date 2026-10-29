from types import SimpleNamespace

import pytest

from tool_loop import ToolLoopError, run_tool_loop
from tools import CALCULATE_TOOL, REGISTRY


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(tool_id, name, tool_input):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)


def reply(blocks, stop_reason):
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


class ScriptedModel:
    """Stands in for client.messages.create and replays prepared replies in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, **kwargs):
        # copy the list, because the loop keeps appending to the original
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.replies.pop(0)


def run_one_tool_call(name, tool_input, registry=REGISTRY):
    first = reply([tool_block("toolu_01", name, tool_input)], "tool_use")
    final = reply([text_block("done")], "end_turn")
    model = ScriptedModel([first, final])
    result = run_tool_loop(model, "go", [CALCULATE_TOOL], registry)
    sent_results = model.requests[1]["messages"][2]["content"]
    return result, sent_results[0]


# ---------- the loop ----------

def test_loop_runs_the_tool_and_returns_the_final_text():
    first = reply(
        [text_block("I'll calculate that."),
         tool_block("toolu_01", "calculate", {"operation": "multiply", "a": 1234, "b": 5678})],
        "tool_use",
    )
    final = reply([text_block("The answer is 7,006,652.")], "end_turn")
    model = ScriptedModel([first, final])

    result = run_tool_loop(model, "What is 1234 times 5678?", [CALCULATE_TOOL], REGISTRY)

    assert result.text == "The answer is 7,006,652."
    assert result.steps == 2
    assert (result.input_tokens, result.output_tokens) == (200, 40)

    second_request = model.requests[1]
    roles = [message["role"] for message in second_request["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert second_request["messages"][1]["content"] == first.content
    assert second_request["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_01", "content": "7006652"}
    ]
    assert second_request["tools"] == [CALCULATE_TOOL]


def test_every_tool_call_in_one_reply_gets_its_own_result_in_one_message():
    first = reply(
        [tool_block("toolu_01", "calculate", {"operation": "multiply", "a": 2, "b": 3}),
         tool_block("toolu_02", "calculate", {"operation": "add", "a": 10, "b": 5})],
        "tool_use",
    )
    final = reply([text_block("6 and 15.")], "end_turn")
    model = ScriptedModel([first, final])

    run_tool_loop(model, "Two sums please", [CALCULATE_TOOL], REGISTRY)

    results = model.requests[1]["messages"][2]["content"]
    assert [item["tool_use_id"] for item in results] == ["toolu_01", "toolu_02"]
    assert [item["content"] for item in results] == ["6", "15"]


def test_loop_gives_up_after_max_steps():
    endless = reply(
        [tool_block("toolu_01", "calculate", {"operation": "add", "a": 1, "b": 1})],
        "tool_use",
    )
    model = ScriptedModel([endless] * 10)

    with pytest.raises(ToolLoopError):
        run_tool_loop(model, "loop forever", [CALCULATE_TOOL], REGISTRY, max_steps=3)

    assert len(model.requests) == 3


def test_a_question_that_needs_no_tool_returns_after_one_step():
    model = ScriptedModel([reply([text_block("Paris.")], "end_turn")])

    result = run_tool_loop(model, "Capital of France?", [CALCULATE_TOOL], registry={})

    assert result.text == "Paris."
    assert result.steps == 1


def test_a_cut_off_reply_is_returned_with_its_stop_reason():
    model = ScriptedModel([reply([text_block("The answer is")], "max_tokens")])

    result = run_tool_loop(model, "hi", [CALCULATE_TOOL], REGISTRY)

    assert result.stop_reason == "max_tokens"
    assert result.text == "The answer is"


# ---------- tool failures ----------

def test_unknown_tool_becomes_an_error_result_not_a_crash():
    result, sent = run_one_tool_call("get_weather", {"city": "Paris"})
    assert sent["is_error"] is True
    assert "unknown tool 'get_weather'" in sent["content"]
    assert "calculate" in sent["content"]            # lists what is available
    assert result.text == "done"                      # the loop carried on


def test_a_tool_that_raises_becomes_an_error_result():
    result, sent = run_one_tool_call("calculate", {"operation": "divide", "a": 1, "b": 0})
    assert sent["is_error"] is True
    assert "ZeroDivisionError" in sent["content"]


def test_bad_arguments_become_an_error_result():
    result, sent = run_one_tool_call("calculate", {"operation": "add", "a": 1})
    assert sent["is_error"] is True
    assert "TypeError" in sent["content"]


def test_a_successful_result_has_no_error_flag():
    result, sent = run_one_tool_call("calculate", {"operation": "add", "a": 2, "b": 3})
    assert "is_error" not in sent
    assert sent["content"] == "5"