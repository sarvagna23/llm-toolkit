from dataclasses import dataclass
from typing import Any, Callable

ToolFunction = Callable[..., Any]


class ToolLoopError(Exception):
    """The model kept asking for tools and never gave a final answer."""


@dataclass
class ToolRunResult:
    text: str
    stop_reason: str | None
    steps: int
    input_tokens: int
    output_tokens: int
    messages: list[dict[str, Any]]


def execute_tool(block: Any, registry: dict[str, ToolFunction]) -> dict[str, Any]:
    function = registry.get(block.name)
    if function is None:
        known = ", ".join(sorted(registry)) or "none"
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": f"Error: unknown tool {block.name!r}. Available tools: {known}.",
            "is_error": True,
        }

    try:
        output = function(**block.input)
    except Exception as error:                         # the model should see any tool failure
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": f"Error: {type(error).__name__}: {error}",
            "is_error": True,
        }

    return {"type": "tool_result", "tool_use_id": block.id, "content": str(output)}


def run_tool_loop(
    send: Callable[..., Any],
    prompt: str,
    tools: list[dict[str, Any]],
    registry: dict[str, ToolFunction],
    *,
    model: str = "claude-sonnet-5-5",
    max_tokens: int = 1024,
    max_steps: int = 5,
) -> ToolRunResult:
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    input_tokens = 0
    output_tokens = 0

    for step in range(1, max_steps + 1):
        response = send(model=model, max_tokens=max_tokens, tools=tools, messages=messages)
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens

        if response.stop_reason != "tool_use":
            text = "".join(block.text for block in response.content if block.type == "text")
            return ToolRunResult(text, response.stop_reason, step, input_tokens, output_tokens, messages)

        messages.append({"role": "assistant", "content": response.content})

        tool_results: list[dict[str, Any]] = []
        for block in response.content:
            if block.type != "tool_use":
                continue                                  # skip text blocks
            tool_results.append(execute_tool(block, registry))
        messages.append({"role": "user", "content": tool_results})

    raise ToolLoopError(f"no final answer after {max_steps} steps")