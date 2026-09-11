"""Experiment orchestration with an explicit validation-to-benchmark boundary."""
from __future__ import annotations

import json
import platform
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .config import DATASET_REVISION, FEATURE_NAMES, FEATURE_VERSION, MODEL, MODEL_REVISION, SEED, TOP_N, TOKEN_BUDGET, WINDOW_TOKENS, WINDOW_OVERLAP
from .io import digest, read_csv, read_json, read_jsonl, stable_hash, write_csv, write_json, write_jsonl
from .metrics import aggregate, query_metrics
from .retrieval import Encoder, MemoryIndex, add_logistic

FEATURE_FIELDS = ["query_id", "persona_id", "memory_id", *FEATURE_NAMES, "rrf", "label", "token_count"]
METHODS = ["bm25", "semantic", "rrf", "logistic", "logistic_no_position", "random_forest"]
_worker_encoder = None


def _initialize_feature_worker(cache_path: str):
    global _worker_encoder
    _worker_encoder = Encoder(Path(cache_path), threads=2)


def _persona_features(memories: list[dict], queries: list[dict]) -> list[dict]:
    if _worker_encoder is None:
        raise RuntimeError("Feature worker was not initialized")
    index = MemoryIndex(memories, _worker_encoder)
    result = []
    for query in queries:
        for candidate in index.candidates(query["query"]):
            result.append({"query_id": query["query_id"], "persona_id": query["persona_id"],
                           **candidate, "label": int(candidate["memory_id"] in query["gold_memory_ids"])})
    return result


def source_fingerprint(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): digest(p) for directory in ("src/persona_memory_ranker", "R")
            for p in sorted((root/directory).glob("*")) if p.suffix in (".py", ".R")}


def protocol_inputs(root: Path) -> dict:
    model_files = ["logistic.json", "logistic_no_position.json", "models.rds"]
    # R worker may store several RDS files; capture every frozen model artifact.
    model_paths = sorted((root/"artifacts/models").glob("*.rds")) + [root/"artifacts/models"/name for name in model_files[:2]]
    return {"dataset_revision": DATASET_REVISION, "encoder": MODEL, "encoder_revision": MODEL_REVISION,
            "feature_version": FEATURE_VERSION, "feature_names": FEATURE_NAMES, "seed": SEED,
            "top_n": TOP_N, "context_budget": TOKEN_BUDGET, "window_tokens": WINDOW_TOKENS, "window_overlap": WINDOW_OVERLAP,
            "split_audit_sha256": digest(root/"reports/split_audit.json"),
            "memory_sha256": digest(root/"data/processed/memories.jsonl"),
            "queries_sha256": {s: digest(root/f"data/processed/{s}_queries.jsonl") for s in ("train", "val", "benchmark")},
            "development_features_sha256": {s: digest(root/f"data/processed/features_{s}.csv") for s in ("train", "val")},
            "models_sha256": {p.name: digest(p) for p in model_paths}, "source_sha256": source_fingerprint(root)}


def verify_frozen(root: Path) -> dict:
    freeze = read_json(root/"artifacts/frozen_protocol.json")
    current = protocol_inputs(root)
    if current != freeze["inputs"]:
        raise ValueError("Frozen protocol inputs changed. Benchmark scoring is blocked; inspect provenance before a new experiment.")
    return freeze


def evaluation_inputs(root: Path, split: str) -> dict:
    """Identify all bytes used to generate reported rankings and labels."""
    source_split = "train" if split == "cv" else split
    paths = [root/f"data/processed/features_{source_split}.csv",
             root/f"data/processed/{source_split}_queries.jsonl",
             root/"data/processed/memories.jsonl", root/"data/download_manifest.json",
             root/"reports/split_audit.json",
             root/f"artifacts/models/{split}_predictions.csv",
             root/"artifacts/models/logistic.json", root/"artifacts/models/logistic_no_position.json"]
    paths += sorted((root/"artifacts/models").glob("*.rds"))
    if split == "cv":
        paths += [root/"artifacts/models/cv_assignments.csv", root/"artifacts/models/training_report.json"]
    return {"files": {p.relative_to(root).as_posix(): digest(p) for p in paths},
            "source": source_fingerprint(root)}


def verify_evaluation(root: Path, split: str, result: dict | None = None) -> dict:
    result = result or read_json(root/f"reports/{split}_evaluation.json")
    if result.get("input_sha256") != evaluation_inputs(root, split):
        raise ValueError(f"Stale {split} evaluation: input data, predictions, models or code changed")
    outputs = result.get("output_sha256", {})
    expected = {f"reports/{split}_query_metrics.csv", f"reports/{split}_summary.csv"}
    allowed = expected | {f"reports/{split}_updates.csv"}
    if not expected <= set(outputs) <= allowed or any(digest(root/path) != value for path, value in outputs.items()):
        raise ValueError(f"Stale {split} evaluation: metric tables differ from the accepted evaluation")
    return result


def build_features(root: Path, split: str, workers: int = 3) -> dict:
    if split == "benchmark":
        verify_frozen(root)
    queries = read_jsonl(root/f"data/processed/{split}_queries.jsonl")
    by_persona = defaultdict(list)
    for query in queries:
        by_persona[query["persona_id"]].append(query)
    memory_groups = defaultdict(list)
    for memory in read_jsonl(root/"data/processed/memories.jsonl"):
        if memory["persona_id"] in by_persona:
            memory_groups[memory["persona_id"]].append(memory)
    if workers < 1 or workers > 3:
        raise ValueError("Feature workers must be between 1 and 3")
    if set(by_persona) != set(memory_groups):
        raise ValueError("Query has no history")
    results = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_initialize_feature_worker,
                             initargs=(str(root/"data/cache/embeddings"),)) as pool:
        futures = {pool.submit(_persona_features, memory_groups[pid], by_persona[pid]): pid for pid in sorted(by_persona)}
        for number, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            if number % 10 == 0 or number == len(by_persona):
                print(f"{split}: encoded {number}/{len(by_persona)} personas", flush=True)
    rows = [row for pid in sorted(results) for row in results[pid]]
    write_csv(root/f"data/processed/features_{split}.csv", rows, FEATURE_FIELDS)
    write_json(root/f"reports/features_{split}.json", {"split": split, "queries": len(queries), "personas": len(by_persona),
                "candidate_pairs": len(rows), "positives": sum(r["label"] for r in rows), "feature_version": FEATURE_VERSION,
                "encoder": MODEL, "encoder_revision": MODEL_REVISION, "source_query_sha256": digest(root/f"data/processed/{split}_queries.jsonl")})
    return {"split": split, "queries": len(queries), "candidate_pairs": len(rows)}


def load_scored(root: Path, split: str, prediction_path: Path | None = None) -> tuple[list[dict], list[dict], float | None]:
    source_split = "train" if split == "cv" else split
    rows = read_csv(root/f"data/processed/features_{source_split}.csv")
    for row in rows:
        for field in [*FEATURE_NAMES, "rrf", "label", "token_count"]:
            row[field] = float(row[field])
    queries = read_jsonl(root/f"data/processed/{source_split}_queries.jsonl")
    if split != "cv":
        add_logistic(rows, root/"artifacts/models")
    path = prediction_path or root/f"artifacts/models/{split}_predictions.csv"
    predictions = read_csv(path)
    for prediction in predictions:
        for name in ("logistic", "logistic_no_position", "random_forest"):
            value = float(prediction[name])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid imported R prediction: {name}")
            prediction[name] = value
    lookup = {(p["query_id"], p["memory_id"]): p for p in predictions}
    keys = {(r["query_id"], r["memory_id"]) for r in rows}
    if len(lookup) != len(predictions) or len(keys) != len(rows) or set(lookup) != keys:
        raise ValueError("R prediction keys differ from candidate features")
    if split == "cv":
        validate_oof(root, rows, predictions)
    max_error = None if split == "cv" else 0.0
    for row in rows:
        prediction = lookup[(row["query_id"], row["memory_id"])]
        for name in ("logistic", "logistic_no_position"):
            if split == "cv":
                row[name] = prediction[name]
            else:
                error = abs(row[name] - prediction[name])
                max_error = max(max_error, error)
        row["random_forest"] = float(prediction["random_forest"])
    if max_error is not None and max_error > 1e-8:
        raise ValueError(f"R/Python logistic prediction mismatch: {max_error}")
    if any(not np.isfinite([r[k] for k in METHODS]).all() for r in rows):
        raise ValueError("Nonfinite model predictions")
    return rows, queries, max_error


def validate_oof(root: Path, rows: list[dict], predictions: list[dict]) -> None:
    assignments = read_csv(root/"artifacts/models/cv_assignments.csv")
    folds = {r["persona_id"]: int(r["fold"]) for r in assignments}
    personas = {r["persona_id"] for r in rows}
    if len(folds) != len(assignments) or set(folds) != personas:
        raise ValueError("OOF persona assignments do not cover the training population")
    report = read_json(root/"artifacts/models/training_report.json")
    reported = {int(f["fold"]): f for f in report["folds"]}
    if len(reported) != len(report["folds"]) or set(reported) != set(folds.values()):
        raise ValueError("OOF training report fold assignments differ")
    for number, fold in reported.items():
        heldout = {pid for pid in personas if folds[pid] == number}
        if set(fold["heldout_personas"]) != heldout or set(fold["train_personas"]) != personas-heldout:
            raise ValueError("OOF training and held-out personas overlap or are incomplete")
    ids = {(r["query_id"], r["memory_id"]): r["persona_id"] for r in rows}
    for prediction in predictions:
        pid = ids[(prediction["query_id"], prediction["memory_id"])]
        if prediction["persona_id"] != pid or int(prediction["fold"]) != folds[pid]:
            raise ValueError("OOF prediction has the wrong held-out persona or fold")


def evaluate(root: Path, split: str) -> dict:
    freeze = verify_frozen(root) if split == "benchmark" else None
    inputs = evaluation_inputs(root, split)
    rows, queries, error = load_scored(root, split)
    groups = defaultdict(list)
    for row in rows:
        groups[row["query_id"]].append(row)
    metrics = []
    for q in queries:
        for method in METHODS:
            metrics.append({"query_id": q["query_id"], "persona_id": q["persona_id"], "split": split,
                            "updated": q["updated"], "method": method,
                            **query_metrics(groups[q["query_id"]], q["gold_memory_ids"], method)})
    write_csv(root/f"reports/{split}_query_metrics.csv", metrics)
    summary = aggregate(metrics)
    write_csv(root/f"reports/{split}_summary.csv", summary)
    groups_summary = []
    for updated in ("true", "false"):
        subset = [r for r in metrics if str(r["updated"]).lower() == updated]
        if subset:
            groups_summary.extend(dict(updated=updated, **r) for r in aggregate(subset))
    if groups_summary:
        write_csv(root/f"reports/{split}_updates.csv", groups_summary)
    if evaluation_inputs(root, split) != inputs:
        raise ValueError("Evaluation inputs changed during scoring; no summary is accepted")
    output_paths = [root/f"reports/{split}_query_metrics.csv", root/f"reports/{split}_summary.csv"]
    if groups_summary:
        output_paths.append(root/f"reports/{split}_updates.csv")
    result = {"split": split, "queries": len(queries), "r_python_max_error": error, "input_sha256": inputs,
              "output_sha256": {p.relative_to(root).as_posix(): digest(p) for p in output_paths},
              "synthetic_data": True, "persona_limit_per_split": read_json(root/"data/download_manifest.json").get("persona_limit_per_split"),
              "summary": summary, "protocol_sha256": stable_hash(freeze) if freeze else None}
    write_json(root/f"reports/{split}_evaluation.json", result)
    return {k: v for k, v in result.items() if k not in ("summary", "input_sha256", "output_sha256")}


def freeze_protocol(root: Path) -> dict:
    target = root/"artifacts/frozen_protocol.json"
    if target.exists():
        raise ValueError("Protocol already frozen; do not overwrite an evaluated experiment")
    evaluation = verify_evaluation(root, "val")
    if evaluation.get("persona_limit_per_split") is not None:
        raise ValueError("Development subset must not unlock official benchmark scoring")
    eligible = {r["method"]: r["mean"] for r in evaluation["summary"] if r["metric"] == "budget_recall" and r["method"] in ("bm25", "semantic", "rrf", "logistic")}
    if len(eligible) != 4:
        raise ValueError("Missing validation baseline or portable model")
    # Baseline wins an exact tie. RF remains an experimental R comparison model.
    priority = ["rrf", "semantic", "bm25", "logistic"]
    selected = max(priority, key=lambda method: eligible[method])
    manifest = {"schema_version": 1, "selection_metric": "validation mean budget_recall",
                "selected_portable_method": selected, "validation_scores": eligible,
                "inputs": protocol_inputs(root), "python": platform.python_version(),
                "human_audit_status": "pending; 100-case queue is not a completed human review"}
    write_json(target, manifest)
    write_json(root/"artifacts/models/selection.json", {"schema_version": 1, "feature_version": FEATURE_VERSION,
               "selected_method": selected, "selection_metric": manifest["selection_metric"],
               "protocol_sha256": stable_hash(manifest)})
    return {"selected_portable_method": selected, "protocol_sha256": stable_hash(manifest)}


def audit_queue(root: Path) -> dict:
    queue_path = root/"reports/private/human_audit_100.csv"
    review_fields = ("alignment_correct", "other_supporting_memory_ids", "notes", "reviewer", "reviewed_at")
    if queue_path.exists() and any(str(row.get(field, "")).strip()
                                  for row in read_csv(queue_path) for field in review_fields):
        raise ValueError("Audit queue contains reviewer input; preserve it and use a separate experiment directory")
    queries = read_jsonl(root/"data/processed/train_queries.jsonl") + read_jsonl(root/"data/processed/val_queries.jsonl")
    queries = sorted(queries, key=lambda q: stable_hash([SEED, "human-audit", q["query_id"]]))[:100]
    memories = {m["memory_id"]: m for m in read_jsonl(root/"data/processed/memories.jsonl")}
    wanted = {q["query_id"] for q in queries}
    candidates = defaultdict(list)
    import csv
    for split in ("train", "val"):
        with (root/f"data/processed/features_{split}.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["query_id"] in wanted:
                    candidates[row["query_id"]].append(row)
    alternatives = {}
    for qid, rows_for_query in candidates.items():
        ids = set()
        for method in ("bm25", "semantic", "rrf"):
            ids.update(r["memory_id"] for r in sorted(rows_for_query, key=lambda r: (-float(r[method]), r["memory_id"]))[:5])
        alternatives[qid] = [memories[mid] for mid in sorted(ids)]
    rows = [{"query_id": q["query_id"], "persona_id": q["persona_id"], "split": q["split"], "query": q["query"],
             "gold_memory_ids": json.dumps(q["gold_memory_ids"]),
             "source_evidence": json.dumps([memories[mid] for mid in q["gold_memory_ids"]], ensure_ascii=False),
             "top_candidate_evidence": json.dumps(alternatives.get(q["query_id"], []), ensure_ascii=False),
             "alignment_correct": "", "other_supporting_memory_ids": "", "notes": "", "reviewer": "", "reviewed_at": ""} for q in queries]
    write_csv(root/"reports/private/human_audit_100.csv", rows)
    write_json(root/"reports/human_audit_status.json", {"queued": len(rows), "completed": 0, "status": "pending human review"})
    return {"queued": len(rows), "completed": 0}
