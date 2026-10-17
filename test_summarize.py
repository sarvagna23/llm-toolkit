import asyncio
import json
from pathlib import Path

import pytest

from client import Completion
from pricing import CostTracker
from summarize import (
    SourceDoc,
    build_prompt,
    error_hint,
    find_files,
    format_report,
    read_for_summary,
    summarize_docs,
    write_jsonl,
)


def make_docs(tmp_path):
    (tmp_path / "b.md").write_text("# B\nbody")
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "UPPER.TXT").write_text("loud")
    (tmp_path / "empty.txt").write_text("")
    (tmp_path / "skip.pdf").write_text("nope")
    (tmp_path / ".hidden.txt").write_text("secret")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.txt").write_text("deep")


def test_find_files_filters_and_sorts(tmp_path):
    make_docs(tmp_path)
    names = [path.name for path in find_files(tmp_path)]
    assert names == ["UPPER.TXT", "a.txt", "b.md", "empty.txt"]


def test_find_files_rejects_a_missing_folder(tmp_path):
    with pytest.raises(NotADirectoryError):
        find_files(tmp_path / "nope")


def test_read_skips_whitespace_only_files(tmp_path):
    (tmp_path / "blank.txt").write_text("  \n\n ")
    assert read_for_summary(tmp_path / "blank.txt", max_chars=100) is None


def test_read_truncates_long_text_and_flags_it(tmp_path):
    (tmp_path / "long.txt").write_text("x" * 50)
    doc = read_for_summary(tmp_path / "long.txt", max_chars=10)
    assert doc is not None
    assert doc.text == "x" * 10
    assert doc.truncated is True


def test_read_keeps_short_text_whole(tmp_path):
    (tmp_path / "short.txt").write_text("  hello  \n")
    doc = read_for_summary(tmp_path / "short.txt", max_chars=10)
    assert doc is not None
    assert doc.text == "hello"
    assert doc.truncated is False


def test_read_survives_bad_bytes(tmp_path):
    (tmp_path / "bad.txt").write_bytes(b"ok \xff\xfe text")
    doc = read_for_summary(tmp_path / "bad.txt", max_chars=100)
    assert doc is not None
    assert "ok" in doc.text


class FakeBatchClient:
    model = "claude-sonnet-5-5"

    def __init__(self, results):
        self._results = results

    async def complete_many(self, prompts, **kwargs):
        assert len(prompts) == len(self._results)
        return list(self._results)


def test_prompt_contains_the_document_text():
    doc = SourceDoc(Path("a.txt"), "alpha beta", False)
    assert "alpha beta" in build_prompt(doc)


def test_summarize_docs_builds_records_in_order():
    docs = [
        SourceDoc(Path("a.txt"), "alpha", False),
        SourceDoc(Path("b.txt"), "beta", True),
        SourceDoc(Path("c.txt"), "gamma", False),
    ]
    results = [
        Completion("sum a", 1_000, 500, 12.0),
        Completion("sum b", 1_000, 500, 0.0, cached=True),
        RuntimeError("boom"),
    ]
    tracker = CostTracker("claude-sonnet-5-5")

    records = asyncio.run(summarize_docs(docs, FakeBatchClient(results), tracker))

    assert [record["file"] for record in records] == ["a.txt", "b.txt", "c.txt"]
    assert records[0]["summary"] == "sum a"
    assert records[0]["cost_usd"] == pytest.approx(0.007)
    assert records[1]["cached"] is True
    assert records[1]["cost_usd"] == 0.0
    assert records[1]["truncated"] is True
    assert "summary" not in records[2]
    assert "RuntimeError" in records[2]["error"]
    assert (tracker.calls, tracker.cache_hits, tracker.failures) == (1, 1, 1)


def test_write_jsonl_writes_one_json_object_per_line(tmp_path):
    records = [{"file": "a.txt", "summary": "café"}, {"file": "b.txt", "error": "boom"}]
    out = tmp_path / "out.jsonl"

    write_jsonl(records, out)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == records


def test_write_jsonl_creates_a_missing_folder(tmp_path):
    out = tmp_path / "results" / "out.jsonl"
    write_jsonl([{"file": "a.txt"}], out)
    assert out.exists()


def test_format_report_lists_each_file_with_its_status():
    records = [
        {"file": "a.txt", "summary": "s", "truncated": False, "input_tokens": 1000,
         "output_tokens": 500, "cached": False, "cost_usd": 0.007},
        {"file": "long.md", "summary": "s", "truncated": True, "input_tokens": 1000,
         "output_tokens": 500, "cached": True, "cost_usd": 0.0},
        {"file": "bad.txt", "error": "RuntimeError: boom"},
    ]

    lines = format_report(records).splitlines()

    assert len(lines) == 4                       # a header plus three files
    assert "paid" in lines[1] and "$0.0070" in lines[1]
    assert "cached, truncated" in lines[2] and "$0.0000" in lines[2]
    assert "FAILED" in lines[3]


def test_error_hint_recognizes_known_failures():
    auth = [{"file": "a.txt", "error": "AuthenticationError: bad key"}]
    limited = [{"file": "a.txt", "error": "RateLimitError: slow down"}]
    fine = [{"file": "a.txt", "summary": "s"}]
    other = [{"file": "a.txt", "error": "ValueError: weird"}]

    assert "ANTHROPIC_API_KEY" in str(error_hint(auth))
    assert "--concurrency" in str(error_hint(limited))
    assert error_hint(fine) is None
    assert error_hint(other) is None