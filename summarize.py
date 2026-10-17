import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cache import DiskCache
from client import AsyncLLMClient, Completion
from pricing import CostTracker

SUPPORTED_SUFFIXES = {".txt", ".md"}


@dataclass(frozen=True)
class SourceDoc:
    path: Path
    text: str
    truncated: bool          # True if the file was longer than max_chars and got cut


def find_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a folder")

    files = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not path.name.startswith(".")
    ]
    return sorted(files)


def read_for_summary(path: Path, max_chars: int) -> SourceDoc | None:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return None                              # empty files are skipped

    truncated = len(text) > max_chars
    return SourceDoc(path=path, text=text[:max_chars], truncated=truncated)


def load_docs(folder: Path, max_chars: int) -> tuple[list[SourceDoc], list[Path]]:
    docs: list[SourceDoc] = []
    skipped: list[Path] = []
    for path in find_files(folder):
        doc = read_for_summary(path, max_chars)
        if doc is None:
            skipped.append(path)
        else:
            docs.append(doc)
    return docs, skipped


def build_prompt(doc: SourceDoc) -> str:
    return (
        "Summarize the document below in 2 to 3 sentences. "
        "Reply with the summary only.\n\n"
        f"Document:\n{doc.text}"
    )


def make_record(
    doc: SourceDoc,
    result: Completion | BaseException,
    tracker: CostTracker,
) -> dict[str, Any]:
    if isinstance(result, BaseException):
        tracker.record_failure()
        return {"file": doc.path.name, "error": f"{type(result).__name__}: {result}"}

    tracker.record_completion(result)
    return {
        "file": doc.path.name,
        "summary": result.text,
        "truncated": doc.truncated,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cached": result.cached,
        "cost_usd": round(tracker.cost_of(result), 6),
    }


async def summarize_docs(
    docs: list[SourceDoc],
    client: AsyncLLMClient,
    tracker: CostTracker,
    *,
    max_tokens: int = 300,
) -> list[dict[str, Any]]:
    prompts = [build_prompt(doc) for doc in docs]
    results = await client.complete_many(prompts, max_tokens=max_tokens)
    return [
        make_record(doc, result, tracker)
        for doc, result in zip(docs, results, strict=True)
    ]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)     # create the output folder if needed
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_report(records: list[dict[str, Any]]) -> str:
    if not records:
        return "(no files)"

    name_width = max(len(record["file"]) for record in records)
    header = f"{'file':<{name_width}}  {'in':>7}  {'out':>6}  {'cost':>9}  status"
    lines = [header]

    for record in records:
        name = record["file"]
        if "error" in record:
            lines.append(f"{name:<{name_width}}  {'-':>7}  {'-':>6}  {'-':>9}  FAILED")
            continue

        status = "cached" if record["cached"] else "paid"
        if record["truncated"]:
            status += ", truncated"
        cost_text = f"${record['cost_usd']:.4f}"
        lines.append(
            f"{name:<{name_width}}  {record['input_tokens']:>7}  "
            f"{record['output_tokens']:>6}  {cost_text:>9}  {status}"
        )

    return "\n".join(lines)


def error_hint(records: list[dict[str, Any]]) -> str | None:
    error_names = {
        record["error"].split(":", 1)[0]
        for record in records
        if "error" in record
    }
    if "AuthenticationError" in error_names:
        return "The API key was rejected. Check ANTHROPIC_API_KEY."
    if "RateLimitError" in error_names:
        return "Rate limited even after retries. Try a lower --concurrency."
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize every .txt and .md file in a folder")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, default=Path("summaries.jsonl"))
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=20000)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: set ANTHROPIC_API_KEY first", file=sys.stderr)
        sys.exit(1)

    try:
        docs, skipped = load_docs(args.folder, args.max_chars)
    except NotADirectoryError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    for path in skipped:
        print(f"skipped (empty): {path.name}")
    if not docs:
        print(f"No non-empty .txt or .md files in {args.folder}")
        return

    cache = None if args.no_cache else DiskCache()
    client = AsyncLLMClient(max_concurrent=args.concurrency, cache=cache)
    tracker = CostTracker(client.model)

    records = asyncio.run(summarize_docs(docs, client, tracker))
    write_jsonl(records, args.out)

    print(format_report(records))
    print()
    print(tracker.summary())
    print(f"results written to {args.out}")

    hint = error_hint(records)
    if hint:
        print(f"Hint: {hint}", file=sys.stderr)
    if tracker.failures:
        sys.exit(1)                      # exit code 1 means "something failed"


if __name__ == "__main__":
    main()