from dataclasses import dataclass

from client import Completion


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float     # dollars per million input tokens
    output_per_million: float    # dollars per million output tokens


# from platform.claude.com/docs/en/about-claude/pricing, checked 2026-10-17
PRICES: dict[str, ModelPrice] = {
    "claude-sonnet-5-5": ModelPrice(input_per_million=2.0, output_per_million=10.0),
}


def price_for(model: str) -> ModelPrice:
    try:
        return PRICES[model]
    except KeyError:
        raise ValueError(f"No price known for model {model!r}. Add it to PRICES.") from None


def cost_usd(input_tokens: int, output_tokens: int, price: ModelPrice) -> float:
    input_cost = input_tokens / 1_000_000 * price.input_per_million
    output_cost = output_tokens / 1_000_000 * price.output_per_million
    return input_cost + output_cost


class CostTracker:
    def __init__(self, model: str):
        self.model = model
        self.price = price_for(model)
        self.calls = 0
        self.cache_hits = 0
        self.failures = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.saved_usd = 0.0          # what the cache hits would have cost

    def record(self, input_tokens: int, output_tokens: int, *, cached: bool) -> None:
        if cached:
            self.cache_hits += 1
            self.saved_usd += cost_usd(input_tokens, output_tokens, self.price)
            return
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def record_completion(self, completion: Completion) -> None:
        self.record(completion.input_tokens, completion.output_tokens, cached=completion.cached)

    def record_failure(self) -> None:
        self.failures += 1

    def cost_of(self, completion: Completion) -> float:
        if completion.cached:
            return 0.0                      # a cache hit costs nothing
        return cost_usd(completion.input_tokens, completion.output_tokens, self.price)

    @property
    def total_usd(self) -> float:
        return cost_usd(self.input_tokens, self.output_tokens, self.price)

    def summary(self) -> str:
        return (
            f"calls: {self.calls} paid, {self.cache_hits} from cache, {self.failures} failed\n"
            f"tokens: {self.input_tokens} in, {self.output_tokens} out\n"
            f"cost: ${self.total_usd:.4f} (cache saved about ${self.saved_usd:.4f})"
        )