# LLM batch summarizer

Summarizes every `.txt` and `.md` file in a folder, concurrently, with retries,
a disk cache, and a cost report. Built on a small typed client (`client.py`).

## Setup
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
```

## Usage
```
python summarize.py docs/ --out summaries.jsonl --concurrency 5 --max-chars 20000
```

| Flag | Default | Meaning |
|---|---|---|
| `folder` | required | folder holding the files |
| `--out` | `summaries.jsonl` | results file |
| `--concurrency` | 5 | most calls in flight at once |
| `--max-chars` | 20000 | longer files are cut to this many characters |
| `--no-cache` | off | ignore and don't write the disk cache |

## How it works
```
folder -> find files -> read and truncate -> async client (cache, retries, concurrency limit)
       -> records -> summaries.jsonl + cost report
```

- **Async client with a semaphore:** many requests at once, never more than `--concurrency`.
- **Retries:** rate limits, server errors, and connection problems are retried with
  exponential backoff and jitter. Permanent errors, such as a rejected key, are not.
- **Disk cache:** keyed by a hash of the full request (model, settings, prompt), so a
  repeat run reuses saved answers.
- **Failure handling:** one failed file doesn't stop the batch. Failed files are listed,
  and the exit code is 1.

## Output
One JSON object per line:
```
{"file": "notes.md", "summary": "...", "truncated": false, "input_tokens": 812, "output_tokens": 96, "cached": false, "cost_usd": 0.0039}
```
Failed files have an `"error"` field instead of `"summary"`.

A table is printed at the end, with one row per file:
```
file            in     out       cost  status
cats.txt         -       -          -  FAILED
```
(That sample shows a failed row, from a run with a rejected key.)

## Tests
```
pytest -v
mypy client.py pricing.py summarize.py
```
The tests use a fake API, so they need no key and cost nothing. They cover file
discovery, truncation, retries, the concurrency limit, the cache, cost math, and
record order.

## Limits
- Only the top level of the folder is read, not subfolders.
- File text is placed directly in the prompt. Only summarize files you trust.
- Two identical prompts sent at the same moment both reach the API, because the cache
  is only filled after a call finishes.
- Cost uses the Claude Sonnet 5.5 prices in `pricing.py`, checked on 2026-10-17.
  Prices change, so check them before relying on the numbers.