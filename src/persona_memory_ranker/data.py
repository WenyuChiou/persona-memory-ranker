"""Public dataset ingestion and source-preserving evidence alignment."""
from __future__ import annotations

import ast
import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

from .config import DATASET, DATASET_REVISION, SEED, SPLITS
from .io import download, read_csv, read_json, stable_hash, write_csv, write_json, write_jsonl


def structured(value: str):
    """The published CSV mixes JSON and Python literals; never execute a cell."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def checked_history_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError("Unsafe history path")
    if len(path.parts) != 3 or path.parts[:2] != ("data", "chat_history_32k") or path.suffix != ".json":
        raise ValueError(f"Unexpected history path: {value}")
    return str(path)


def ingest(root: Path, persona_limit: int | None = None) -> dict:
    """Download split tables and selected histories; limit is a labelled development subset."""
    csv.field_size_limit(10_000_000)
    raw = root / "data/raw"
    manifest_path = root / "data/download_manifest.json"
    old = read_json(manifest_path) if manifest_path.exists() else {"files": []}
    if old.get("revision", DATASET_REVISION) != DATASET_REVISION:
        raise ValueError("Existing manifest uses a different dataset revision")
    known = {f["path"]: f for f in old["files"]}
    files = dict(known)
    base = f"https://huggingface.co/datasets/{DATASET}/resolve/{DATASET_REVISION}/"

    def fetch(relative):
        destination = raw / relative
        key = destination.relative_to(root).as_posix()
        result = download(base + relative, destination, known.get(key, {}).get("sha256"))
        result["path"] = key
        return result

    for name in ["README.md", *(f"benchmark/text/{s if s != 'benchmark' else 'benchmark'}.csv" for s in SPLITS)]:
        item = fetch(name)
        files[item["path"]] = item
    # Persist verified tables before schema checks, so a data failure never forces a download.
    write_json(manifest_path, {"revision": DATASET_REVISION, "files": list(files.values())})
    sources = {split: read_csv(raw / f"benchmark/text/{split}.csv") for split in SPLITS}
    official_ids = {s: {r["persona_id"] for r in rows} for s, rows in sources.items()}
    overlaps = {f"{a}/{b}": sorted(official_ids[a] & official_ids[b])
                for a, b in (("train", "val"), ("train", "benchmark"), ("val", "benchmark"))}
    # The pinned release has train/val persona overlap. Keep the benchmark untouched.
    # This explicit repair is fixed before any scoring and recorded in the public audit.
    development = sorted((official_ids["train"] | official_ids["val"]) - official_ids["benchmark"],
                         key=lambda pid: stable_hash([SEED, "development", pid]))
    cutoff = math.floor(.8 * len(development))
    assignment = {pid: "train" if i < cutoff else "val" for i, pid in enumerate(development)}
    assignment.update({pid: "benchmark" for pid in official_ids["benchmark"]})
    selected = {}
    for split in SPLITS:
        ids = sorted([pid for pid, group in assignment.items() if group == split], key=lambda pid: stable_hash([SEED, split, pid]))
        selected[split] = ids if persona_limit is None else ids[:persona_limit]
    rows, errors = [], []
    split_counts = {s: len(source) for s, source in sources.items()}
    for original_split, source in sources.items():
        for index, row in enumerate(source):
            query_id = f"{original_split}-{index:05d}"
            split = assignment[row["persona_id"]]
            if original_split != "benchmark" and split == "benchmark":
                errors.append({"query_id": query_id, "reason": "development_persona_present_in_benchmark"})
                continue
            if row["persona_id"] not in selected[split]:
                continue
            try:
                query = structured(row["user_query"])
                gold = structured(row["related_conversation_snippet"])
                if not isinstance(query, dict) or query.get("role") != "user" or not isinstance(query.get("content"), str):
                    raise ValueError("Invalid user_query shape")
                if not isinstance(gold, list) or not gold or any(not isinstance(m, dict) or not isinstance(m.get("content"), str) or m.get("role") not in ("user", "assistant") for m in gold):
                    raise ValueError("Invalid evidence message list")
                history = checked_history_path(row["chat_history_32k_link"])
                rows.append(dict(query_id=query_id, persona_id=row["persona_id"], split=split,
                                 query=query["content"], history_path=history,
                                 gold_json=json.dumps(gold, ensure_ascii=False), updated=row.get("updated", ""),
                                 who=row.get("who", ""), topic_query=row.get("topic_query", ""), original_split=original_split))
            except (ValueError, SyntaxError, KeyError, TypeError) as exc:
                errors.append({"query_id": query_id, "reason": str(exc)})
    paths = sorted({r["history_path"] for r in rows})
    manifest = dict(dataset=DATASET, revision=DATASET_REVISION, license="CC-BY-4.0", synthetic=True,
                    persona_limit_per_split=persona_limit, official_query_counts=split_counts,
                    selected_personas=selected, files=list(files.values()))
    write_json(root / "reports/split_audit.json", {"official_query_counts": split_counts,
               "official_persona_counts": {s: len(ids) for s, ids in official_ids.items()}, "official_overlaps": overlaps,
               "repair": "Preserve benchmark; exclude benchmark personas from pooled development; hash-seed310 80/20 persona split.",
               "repaired_persona_counts": {s: sum(v == s for v in assignment.values()) for s in SPLITS},
               "selected_personas": selected, "original_split_retained": True})
    write_json(manifest_path, manifest)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(fetch, p): p for p in paths}
        for index, future in enumerate(as_completed(pending), 1):
            item = future.result()  # A missing history aborts; never silently skip it.
            files[item["path"]] = item
            if index % 20 == 0 or index == len(paths):
                manifest["files"] = sorted(files.values(), key=lambda f: f["path"])
                write_json(manifest_path, manifest)
                print(f"Verified histories {index}/{len(paths)}", flush=True)
    write_csv(root / "data/intermediate/queries.csv", rows,
              ["query_id", "persona_id", "split", "query", "history_path", "gold_json", "updated", "who", "topic_query", "original_split"])
    write_json(root / "reports/ingest.json", {"rows": len(rows), "excluded": errors, "persona_limit_per_split": persona_limit})
    return {"queries": len(rows), "histories": len(paths), "excluded": len(errors), "subset": persona_limit is not None}


def make_memories(messages: list[dict], persona_id: str, history_path: str) -> list[dict]:
    """A unit begins at a user message and includes following assistant messages."""
    units = []
    for index, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "assistant" and message.get("content") is None:
            # Known source defect: persona731 has a role-only assistant record.
            # Keep original indices; the caller records every such exclusion.
            continue
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant", "system") or not isinstance(message.get("content"), str):
            raise ValueError(f"Malformed history message at {index}")
        # Some released histories prepend the complete synthetic persona. This is
        # privileged generation context, not a user memory, and would leak labels.
        if message["role"] == "system":
            continue
        if message["role"] == "user" or not units:
            units.append({"memory_id": f"p{persona_id}:t{index:05d}", "persona_id": persona_id,
                          "source_ref": history_path, "turn_index": len(units), "message_indices": [], "messages": []})
        units[-1]["message_indices"].append(index)
        units[-1]["messages"].append({"role": message["role"], "content": message["content"]})
    for unit in units:
        unit["text"] = "\n\n".join(f"{m['role']}: {m['content']}" for m in unit["messages"])
    return units


def align_gold(messages: list[dict], gold: list[dict], memories: list[dict]) -> tuple[list[str], str]:
    """Require a unique exact contiguous evidence sequence; no fuzzy label repair."""
    def key(m):
        return m["role"], m.get("content")
    sequence = [key(m) for m in messages]
    target = [key(m) for m in gold]
    matches = [i for i in range(len(sequence) - len(target) + 1) if sequence[i:i+len(target)] == target]
    if len(matches) != 1:
        return [], "missing_exact_sequence" if not matches else "ambiguous_duplicate_sequence"
    indices = set(range(matches[0], matches[0] + len(target)))
    ids = [m["memory_id"] for m in memories if indices.intersection(m["message_indices"])]
    return ids, "aligned"


def prepare_memories(root: Path) -> dict:
    queries = read_csv(root / "data/processed/queries.csv")
    by_path = {}
    history_hashes = {}
    memories = []
    excluded_system_messages = 0
    malformed_messages = []
    for q in queries:
        path = checked_history_path(q["history_path"])
        if path in by_path:
            if by_path[path][2] != q["persona_id"] or by_path[path][3] != q["split"]:
                raise ValueError("History reused across personas or splits")
            continue
        doc = read_json(root / "data/raw" / path)
        messages = doc["chat_history"]
        malformed_messages.extend({"history_path": path, "message_index": i, "reason": "assistant_content_missing"}
                                  for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content") is None)
        if str(doc.get("metadata", {}).get("persona_id")) != q["persona_id"]:
            raise ValueError("History metadata persona mismatch")
        fingerprint = stable_hash([m for m in messages if m.get("role") != "system"])
        if fingerprint in history_hashes and history_hashes[fingerprint] != q["split"]:
            raise ValueError("Identical history crosses splits")
        history_hashes[fingerprint] = q["split"]
        units = make_memories(messages, q["persona_id"], path)
        excluded_system_messages += sum(m["role"] == "system" for m in messages)
        if not units:
            raise ValueError("Empty conversation history")
        by_path[path] = messages, units, q["persona_id"], q["split"]
        memories.extend(units)
    accepted, alignment = [], []
    for q in queries:
        messages, units, _, _ = by_path[q["history_path"]]
        ids, status = align_gold(messages, json.loads(q["gold_json"]), units)
        alignment.append({"query_id": q["query_id"], "persona_id": q["persona_id"], "split": q["split"],
                          "status": status, "gold_count": len(ids), "memory_count": len(units),
                          "updated": q["updated"], "topic_query": q["topic_query"],
                          "gold_relative_position": max((m["turn_index"] / max(1, len(units)-1) for m in units if m["memory_id"] in ids), default="")})
        if status == "aligned":
            accepted.append({k: q[k] for k in ("query_id", "persona_id", "split", "query", "history_path", "updated", "who", "topic_query")} | {"gold_memory_ids": ids})
    write_jsonl(root / "data/processed/memories.jsonl", memories)
    for split in SPLITS:
        write_jsonl(root / f"data/processed/{split}_queries.jsonl", [q for q in accepted if q["split"] == split])
    write_csv(root / "reports/alignment.csv", alignment)
    write_json(root / "reports/history_quality.json", {"excluded_system_messages": excluded_system_messages,
                "excluded_malformed_messages": malformed_messages})
    summary = {"input_queries": len(queries), "aligned_queries": len(accepted), "excluded_queries": len(queries)-len(accepted),
               "memories": len(memories), "histories": len(by_path), "excluded_system_messages": excluded_system_messages,
               "method": "unique exact contiguous role/content sequence"}
    write_json(root / "reports/alignment.json", summary)
    return summary
