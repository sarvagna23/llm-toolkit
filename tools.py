from typing import Any, Callable

CALCULATE_TOOL: dict[str, Any] = {
    "name": "calculate",
    "description": (
        "Does exact arithmetic on two numbers. Use it for any addition, subtraction, "
        "multiplication or division where the exact result matters, especially with large "
        "numbers, because language models make arithmetic mistakes. It returns the result as "
        "a plain number, or an error message if you divide by zero. It handles only two "
        "numbers at a time, so chain several calls for a longer expression."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "Which arithmetic operation to apply to a and b.",
            },
            "a": {"type": "number", "description": "The first number."},
            "b": {"type": "number", "description": "The second number."},
        },
        "required": ["operation", "a", "b"],
        "additionalProperties": False,
    },
}

READ_DOCUMENT_TOOL: dict[str, Any] = {
    "name": "read_document",
    "description": (
        "Returns the full text of one document from the user's docs folder. Use it when the "
        "user asks what a file says, or asks you to summarize or compare files. Give the file "
        "name including its extension, for example cats.txt. It returns the text, or an error "
        "message if the file does not exist or is empty. It cannot read files outside the "
        "docs folder, and it does not list which files exist."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The file name including its extension, for example cats.txt.",
            },
        },
        "required": ["name"],
        "additionalProperties": False,
    },
}

TOOL_DEFINITIONS: list[dict[str, Any]] = [CALCULATE_TOOL, READ_DOCUMENT_TOOL]


def calculate(operation: str, a: float, b: float) -> float:
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        return a / b        # raises ZeroDivisionError when b is 0, and execute_tool reports it
    raise ValueError(f"unknown operation {operation!r}")


# only calculate is registered so far; the Build adds read_document
REGISTRY: dict[str, Callable[..., Any]] = {"calculate": calculate}