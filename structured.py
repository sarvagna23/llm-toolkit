from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import anthropic
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """The model's reply could not be turned into the requested model."""

    def __init__(self, message: str, *, stop_reason: str | None = None, raw_text: str = ""):
        super().__init__(message)
        self.stop_reason = stop_reason
        self.raw_text = raw_text


@dataclass(frozen=True)
class StructuredCompletion(Generic[T]):
    data: T
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cached: bool = False


def schema_for(response_model: type[BaseModel]) -> dict[str, Any]:
    # the SDK moves constraints the API can't enforce (like minimum) into field descriptions
    return anthropic.transform_schema(response_model)