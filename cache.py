import hashlib
import json
from pathlib import Path
from typing import Any


def make_cache_key(request: dict[str, Any]) -> str:
    canonical_text = json.dumps(request, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, directory: str = ".llm_cache"):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=True)       # create the folder if it isn't there

    def _path_for(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path_for(key)
        if not path.exists():
            return None                            # a miss
        try:
            return json.loads(path.read_text())    # a hit
        except json.JSONDecodeError:
            return None                            # a damaged file counts as a miss

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._path_for(key).write_text(json.dumps(value))