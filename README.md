# llm-toolkit

Small, typed toolkit for calling an LLM: a `Completion` dataclass and an `LLMClient`.

## Setup
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
```

## Run
```
python try_client.py
pytest -v
mypy client.py
```