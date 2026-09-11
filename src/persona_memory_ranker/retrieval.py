"""Gold-blind candidate features and portable inference."""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .config import FEATURE_NAMES, FEATURE_VERSION, MODEL, MODEL_REVISION, RRF_K, TOP_N, WINDOW_OVERLAP, WINDOW_TOKENS
from .io import read_json, stable_hash, write_json


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def render_memory(memory: dict) -> str:
    return f"[{memory['memory_id']} | {memory['source_ref']}]\n{memory['text']}"


def validate_memories(memories: list[dict]) -> None:
    seen = set()
    sources = set()
    for memory in memories:
        for field in ("memory_id", "text", "source_ref"):
            if not isinstance(memory.get(field), str) or not memory[field].strip():
                raise ValueError(f"Memory requires nonempty {field}")
        if memory["memory_id"] in seen:
            raise ValueError(f"Duplicate memory_id: {memory['memory_id']}")
        if type(memory.get("turn_index")) is not int or memory["turn_index"] < 0:
            raise ValueError("turn_index must be a nonnegative integer")
        source = (memory["source_ref"], memory["turn_index"])
        if source in sources:
            raise ValueError("Duplicate source turn under different memory IDs")
        sources.add(source)
        seen.add(memory["memory_id"])


class Encoder:
    """Fixed local encoder; windows retain original text, including trailing content."""
    def __init__(self, cache_dir: Path, threads: int = 4):
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(threads)
        self.model = SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cpu")
        self.tokenizer = self.model.tokenizer
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False))

    def windows(self, text: str) -> list[str]:
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False, verbose=False)
        offsets = encoded["offset_mapping"]
        if not offsets:
            return [text]
        result = []
        for start in range(0, len(offsets), WINDOW_TOKENS - WINDOW_OVERLAP):
            stop = min(len(offsets), start + WINDOW_TOKENS)
            left = 0 if start == 0 else offsets[start][0]
            right = len(text) if stop == len(offsets) else offsets[stop-1][1]
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
        value = self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        if value.shape != (len(texts), 384) or not np.isfinite(value).all():
            raise ValueError("Encoder returned invalid shape or nonfinite values")
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
        temporary.replace(path)
        return value


class MemoryIndex:
    def __init__(self, memories: list[dict], encoder: Encoder):
        validate_memories(memories)
        self.memories = memories
        self.encoder = encoder
        self.tokens = [words(m["text"]) for m in memories]
        self.tf = [Counter(t) for t in self.tokens]
        self.lengths = np.array([len(t) for t in self.tokens], dtype=float)
        self.avg_length = float(self.lengths.mean()) if len(memories) else 1.0
        self.df = Counter(t for counts in self.tf for t in counts)
        self.token_counts = [encoder.count(render_memory(m)) for m in memories]
        windows, owners = [], []
        for index, memory in enumerate(memories):
            chunks = encoder.windows(memory["text"])
            windows.extend(chunks)
            owners.extend([index] * len(chunks))
        self.window_owners = np.array(owners, dtype=int)
        self.window_embeddings = encoder.encode(windows, "memories")

    def lexical(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.memories), dtype=float)
        for token in set(words(query)):
            df = self.df.get(token, 0)
            idf = math.log(1 + (len(self.memories) - df + .5) / (df + .5))
            counts = np.array([row.get(token, 0) for row in self.tf], dtype=float)
            denom = counts + 1.5 * (1 - .75 + .75 * self.lengths / max(self.avg_length, 1))
            scores += idf * counts * 2.5 / denom
        return scores

    def candidates(self, query: str) -> list[dict]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be nonempty text")
        if not self.memories:
            return []
        # Pool long-query windows too; the final portion cannot disappear silently.
        query_vectors = self.encoder.encode(self.encoder.windows(query), "queries")
        window_scores = (self.window_embeddings @ query_vectors.T).max(axis=1)
        semantic = np.full(len(self.memories), -np.inf)
        np.maximum.at(semantic, self.window_owners, window_scores)
        lexical = self.lexical(query)
        ids = [m["memory_id"] for m in self.memories]
        order = lambda scores: sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))
        lexical_order, semantic_order = order(lexical), order(semantic)
        lr = {i: rank + 1 for rank, i in enumerate(lexical_order[:TOP_N])}
        sr = {i: rank + 1 for rank, i in enumerate(semantic_order[:TOP_N])}
        selected = sorted(set(lr) | set(sr), key=lambda i: ids[i])
        query_tokens = set(words(query))
        max_turn = max((m["turn_index"] for m in self.memories), default=0)
        rows = []
        for i in selected:
            rows.append({"memory_id": ids[i], "bm25": float(lexical[i]), "semantic": float(semantic[i]),
                         "overlap": len(query_tokens.intersection(self.tf[i])) / max(1, len(query_tokens)),
                         "log_length": math.log1p(self.token_counts[i]),
                         "position": self.memories[i]["turn_index"] / max(1, max_turn),
                         "rrf": (1/(RRF_K+lr[i]) if i in lr else 0) + (1/(RRF_K+sr[i]) if i in sr else 0),
                         "token_count": self.token_counts[i]})
        return rows


class LogisticScorer:
    """Dependency-light R model export consumer; never infers permission or truth."""
    def __init__(self, model: dict):
        if model.get("schema_version") != 1 or model.get("feature_version") != FEATURE_VERSION or model.get("model_type") != "logistic":
            raise ValueError("Unsupported scorer schema or feature version")
        self.names = model["feature_names"]
        if self.names not in (FEATURE_NAMES, FEATURE_NAMES[:-1]):
            raise ValueError("Unexpected features or feature order")
        self.means, self.scales, self.coefficients = [np.asarray(model[k], dtype=float) for k in ("means", "scales", "coefficients")]
        if any(x.shape != (len(self.names),) or not np.isfinite(x).all() for x in (self.means, self.scales, self.coefficients)):
            raise ValueError("Invalid model coefficient vectors")
        self.intercept = float(model["intercept"])
        if not math.isfinite(self.intercept) or np.any(self.scales <= 0):
            raise ValueError("Invalid model intercept/scales")

    @classmethod
    def load(cls, path: Path):
        return cls(read_json(path))

    def score(self, features: list[dict]) -> list[float]:
        if not features:
            return []
        if any(set(row) != set(self.names) for row in features):
            raise ValueError("Scorer accepts exactly the frozen feature allowlist")
        values = np.array([[row[k] for k in self.names] for row in features], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite scorer input")
        logits = self.intercept + ((values-self.means)/self.scales) @ self.coefficients
        # Stable sigmoid without an overflowing exp in an unused branch.
        out = np.empty_like(logits)
        positive = logits >= 0
        out[positive] = 1/(1+np.exp(-logits[positive]))
        ex = np.exp(logits[~positive])
        out[~positive] = ex/(1+ex)
        return out.tolist()


def add_logistic(rows: list[dict], models_dir: Path) -> None:
    for name in ("logistic", "logistic_no_position"):
        model = LogisticScorer.load(models_dir / f"{name}.json")
        scores = model.score([{k: row[k] for k in model.names} for row in rows])
        for row, score in zip(rows, scores, strict=True):
            row[name] = score
