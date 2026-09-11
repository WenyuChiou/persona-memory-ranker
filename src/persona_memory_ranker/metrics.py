"""Query metrics retain missing candidates in the denominator."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .config import SEED, TOKEN_BUDGET


def ranked(rows: list[dict], method: str) -> list[dict]:
    ids = [r["memory_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate candidate memory ID")
    if any(not math.isfinite(float(r[method])) for r in rows):
        raise ValueError("Nonfinite ranking score")
    return sorted(rows, key=lambda r: (-float(r[method]), r["memory_id"]))


def pack(rows: list[dict], budget: int = TOKEN_BUDGET) -> list[dict]:
    if type(budget) is not int or budget < 0:
        raise ValueError("Budget must be a nonnegative integer")
    selected, used, seen = [], 0, set()
    for row in rows:
        if row["memory_id"] in seen:
            raise ValueError("Duplicate memory in context input")
        seen.add(row["memory_id"])
        count = float(row["token_count"])
        if not math.isfinite(count) or count < 1 or count != int(count):
            raise ValueError("Token count must be a positive integer")
        if used + count <= budget:
            selected.append(row)
            used += int(count)
    return selected


def query_metrics(rows: list[dict], gold: list[str], method: str) -> dict:
    expected = set(gold)
    if not expected:
        raise ValueError("Evidence retrieval metrics require at least one gold unit")
    ordered = ranked(rows, method)
    ids = [r["memory_id"] for r in ordered]
    budget_rows = pack(ordered)
    reciprocal = next((1/i for i, mid in enumerate(ids[:10], 1) if mid in expected), 0.0)
    return {"recall_at_5": len(expected.intersection(ids[:5]))/len(expected), "mrr_at_10": reciprocal,
            "budget_recall": len(expected.intersection(r["memory_id"] for r in budget_rows))/len(expected),
            "candidate_recall": len(expected.intersection(ids))/len(expected),
            "context_tokens": sum(int(r["token_count"]) for r in budget_rows), "candidate_count": len(rows)}


def aggregate(metrics: list[dict], bootstrap_reps: int = 1000) -> list[dict]:
    """Query-weighted means; cluster bootstrap resamples whole personas."""
    grouped = defaultdict(list)
    for row in metrics:
        grouped[row["method"]].append(row)
    results = []
    for method, rows in sorted(grouped.items()):
        personas = sorted({r["persona_id"] for r in rows})
        sums = np.array([[sum(float(r[key]) for r in rows if r["persona_id"] == pid)
                          for key in ("recall_at_5", "mrr_at_10", "budget_recall", "candidate_recall")]
                         for pid in personas])
        counts = np.array([sum(r["persona_id"] == pid for r in rows) for pid in personas])
        rng = np.random.default_rng(SEED)
        indices = rng.integers(0, len(personas), size=(bootstrap_reps, len(personas)))
        boot = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)[:, None]
        for col, key in enumerate(("recall_at_5", "mrr_at_10", "budget_recall", "candidate_recall")):
            lo, hi = np.quantile(boot[:, col], [.025, .975])
            results.append({"method": method, "metric": key, "mean": sums[:, col].sum()/counts.sum(),
                            "ci_low": float(lo), "ci_high": float(hi), "queries": len(rows), "personas": len(personas)})
    return results
