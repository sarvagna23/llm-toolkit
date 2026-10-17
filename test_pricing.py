import pytest

from client import Completion
from pricing import CostTracker, ModelPrice, cost_usd, price_for

PRICE = ModelPrice(input_per_million=3.0, output_per_million=15.0)


def test_one_million_input_tokens():
    assert cost_usd(1_000_000, 0, PRICE) == pytest.approx(3.0)


def test_input_and_output_are_priced_separately():
    assert cost_usd(1_000, 500, PRICE) == pytest.approx(0.0105)


def test_zero_tokens_cost_nothing():
    assert cost_usd(0, 0, PRICE) == 0


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        price_for("not-a-real-model")


def test_tracker_adds_up_paid_calls():
    tracker = CostTracker("claude-sonnet-5-5")
    tracker.record(1_000, 500, cached=False)
    tracker.record(1_000, 500, cached=False)
    assert tracker.calls == 2
    assert tracker.input_tokens == 2_000
    assert tracker.total_usd == pytest.approx(0.014)


def test_cache_hits_cost_nothing_but_count_as_savings():
    tracker = CostTracker("claude-sonnet-5-5")
    tracker.record(1_000, 500, cached=False)
    tracker.record(1_000, 500, cached=True)
    assert tracker.calls == 1
    assert tracker.cache_hits == 1
    assert tracker.total_usd == pytest.approx(0.007)
    assert tracker.saved_usd == pytest.approx(0.007)


def test_record_completion_uses_the_cached_flag():
    tracker = CostTracker("claude-sonnet-5-5")
    tracker.record_completion(Completion("a", 1_000, 500, 10.0))
    tracker.record_completion(Completion("a", 1_000, 500, 0.0, cached=True))
    assert tracker.calls == 1
    assert tracker.cache_hits == 1


def test_cost_of_a_cached_completion_is_zero():
    tracker = CostTracker("claude-sonnet-5-5")
    paid = Completion("a", 1_000, 500, 10.0)
    cached = Completion("a", 1_000, 500, 0.0, cached=True)
    assert tracker.cost_of(paid) == pytest.approx(0.007)
    assert tracker.cost_of(cached) == 0.0


def test_summary_reports_calls_hits_and_failures():
    tracker = CostTracker("claude-sonnet-5-5")
    tracker.record_completion(Completion("a", 1_000, 500, 10.0))
    tracker.record_completion(Completion("a", 1_000, 500, 10.0))
    tracker.record_completion(Completion("a", 1_000, 500, 0.0, cached=True))
    tracker.record_failure()
    text = tracker.summary()
    assert "2 paid" in text
    assert "1 from cache" in text
    assert "1 failed" in text