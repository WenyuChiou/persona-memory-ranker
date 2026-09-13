"""Fixed, windowed MiniLM encoder used by classification and retrieval."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .config import MODEL, MODEL_REVISION, WINDOW_OVERLAP, WINDOW_TOKENS


def stable_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


class Encoder:
    """Encode every part of long text and cache only matching finite matrices."""

    def __init__(self, cache_dir: Path, threads: int = 4):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(threads)
        self.model = SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cpu")
        self.tokenizer = self.model.tokenizer
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(
            text, add_special_tokens=False, truncation=False, verbose=False
        ))

    def windows(self, text: str) -> list[str]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        offsets = encoded["offset_mapping"]
        if not offsets:
            return [text]
        result = []
        for start in range(0, len(offsets), WINDOW_TOKENS - WINDOW_OVERLAP):
            stop = min(len(offsets), start + WINDOW_TOKENS)
            left = 0 if start == 0 else offsets[start][0]
            right = len(text) if stop == len(offsets) else offsets[stop - 1][1]
            result.append(text[left:right])
            if stop == len(offsets):
                break
        return result

    def encode(self, texts: list[str], namespace: str) -> np.ndarray:
        key = stable_hash([MODEL, MODEL_REVISION, texts])
        path = self.cache_dir / f"{namespace}-{key}.npy"
        if path.exists():
            value = np.load(path, allow_pickle=False)
            if value.shape != (len(texts), 384) or not np.isfinite(value).all():
                raise ValueError(f"Invalid embedding cache: {path}")
            return value
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        value = self.model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if value.shape != (len(texts), 384) or not np.isfinite(value).all():
            raise ValueError("Encoder returned invalid shape or nonfinite values")
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
        temporary.replace(path)
        return value
