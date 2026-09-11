"""Explicit, atomic I/O and provenance for reproducible local experiments."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        if not rows:
            raise ValueError("Empty CSV requires explicit field names")
        fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def download(url: str, path: Path, expected_sha: str | None = None) -> dict:
    """Reuse only a verified file. Retry transient network failures at most twice."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and expected_sha:
        if digest(path) != expected_sha:
            raise ValueError(f"Cached file checksum mismatch: {path}")
        return {"url": url, "path": path.as_posix(), "sha256": expected_sha, "bytes": path.stat().st_size}
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "PersonaMemoryRanker/0.1 (research)"})
            with urllib.request.urlopen(request, timeout=90) as response, tmp.open("wb") as output:
                while chunk := response.read(1 << 20):
                    output.write(chunk)
            actual = digest(tmp)
            if expected_sha and actual != expected_sha:
                raise ValueError(f"Downloaded file checksum mismatch: {path}")
            os.replace(tmp, path)
            return {"url": url, "path": path.as_posix(), "sha256": actual, "bytes": path.stat().st_size}
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                raise
            if attempt == 2:
                raise
            print(f"Transient download failure for {path.name}: {type(exc).__name__}; retry {attempt + 1}/2", flush=True)
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable download state")
