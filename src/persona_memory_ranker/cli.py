from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .data import ingest, prepare_memories
from .io import read_json, write_json
from .metrics import pack, ranked
from .pipeline import audit_queue, build_features, evaluate, freeze_protocol, verify_evaluation, verify_frozen
from .retrieval import Encoder, MemoryIndex, add_logistic, render_memory


def rscript() -> str:
    command = shutil.which("Rscript")
    if command:
        return command
    paths = sorted(Path("C:/Program Files/R").glob("R-*/bin/Rscript.exe"), reverse=True)
    if not paths:
        raise FileNotFoundError("Rscript is required; install R and add its bin directory to PATH")
    return str(paths[0])


def run_r(root: Path, script: str, *args: str) -> None:
    subprocess.run([rscript(), str(root/"R"/script), *args], cwd=root, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Persona Memory Ranker: reproducible local evidence retrieval")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--persona-limit", type=int, help="Development subset per split; cannot unlock benchmark")
    commands.add_parser("prepare")
    features = commands.add_parser("features")
    features.add_argument("--split", choices=("train", "val", "benchmark"), required=True)
    features.add_argument("--workers", type=int, default=3, choices=(1,2,3))
    commands.add_parser("train")
    predict = commands.add_parser("predict")
    predict.add_argument("--split", choices=("val", "benchmark"), required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--split", choices=("cv", "val", "benchmark"), required=True)
    verification = commands.add_parser("verify")
    verification.add_argument("--split", choices=("cv", "val", "benchmark"), required=True)
    commands.add_parser("freeze")
    commands.add_parser("audit-queue")
    retrieve = commands.add_parser("retrieve")
    retrieve.add_argument("--input", type=Path, required=True, help="JSON {query, memories:[memory_id,text,source_ref,turn_index]}")
    retrieve.add_argument("--output", type=Path, required=True)
    retrieve.add_argument("--models", type=Path, help="Directory containing portable logistic JSON models")
    retrieve.add_argument("--method", choices=("auto", "bm25", "semantic", "rrf", "logistic"), default="auto")
    retrieve.add_argument("--budget", type=int, default=2000)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    command = args.command
    changes_development = command in ("download", "prepare", "train") or (command == "features" and args.split != "benchmark")
    if changes_development and (root/"artifacts/frozen_protocol.json").exists():
        raise ValueError("This experiment is frozen. Use a separate output checkout for changed development inputs; existing evidence is preserved.")
    if command == "download":
        if args.persona_limit is not None and args.persona_limit < 2:
            parser.error("--persona-limit must be at least 2 for grouped validation")
        result = ingest(root, args.persona_limit)
    elif command == "prepare":
        run_r(root, "prepare.R", "--input", "data/intermediate/queries.csv", "--output", "data/processed/queries.csv", "--report", "reports/cleaning.json")
        result = prepare_memories(root)
    elif command == "features":
        result = build_features(root, args.split, args.workers)
    elif command == "train":
        run_r(root, "train.R", "--train", "data/processed/features_train.csv", "--val", "data/processed/features_val.csv", "--out", "artifacts/models")
        result = {"trained": True, "benchmark_used": False}
    elif command == "predict":
        if args.split == "benchmark":
            verify_frozen(root)
        run_r(root, "predict.R", "--features", f"data/processed/features_{args.split}.csv", "--models", "artifacts/models", "--output", f"artifacts/models/{args.split}_predictions.csv")
        result = {"predicted": args.split}
    elif command == "evaluate":
        result = evaluate(root, args.split)
    elif command == "verify":
        verify_evaluation(root, args.split)
        if args.split == "benchmark":
            verify_frozen(root)
        result = {"verified": args.split}
    elif command == "freeze":
        result = freeze_protocol(root)
    elif command == "audit-queue":
        result = audit_queue(root)
    elif command == "retrieve":
        doc = read_json(args.input)
        if not isinstance(doc, dict) or set(doc) != {"query", "memories"}:
            raise ValueError("Input must contain exactly query and memories; labels are not inference inputs")
        if args.method == "auto":
            selection_path = (args.models or root/"artifacts/models") / "selection.json"
            if selection_path.exists():
                selection = read_json(selection_path)
                if selection.get("schema_version") != 1 or selection.get("feature_version") != "pmr-v1" or selection.get("selected_method") not in ("bm25", "semantic", "rrf", "logistic"):
                    raise ValueError("Invalid portable model selection artifact")
                args.method = selection["selected_method"]
                args.models = args.models or root/"artifacts/models"
            else:
                args.method = "rrf"
        encoder = Encoder(root/"data/cache/embeddings")
        index = MemoryIndex(doc["memories"], encoder)
        rows = index.candidates(doc["query"])
        if args.method == "logistic":
            if args.models is None:
                parser.error("--models is required for logistic retrieval")
            add_logistic(rows, args.models)
        chosen = pack(ranked(rows, args.method), args.budget)
        memories = {m["memory_id"]: m for m in doc["memories"]}
        evidence = [memories[r["memory_id"]] | {"score": r[args.method]} for r in chosen]
        context = "\n\n".join(render_memory(m) for m in evidence)
        token_count = encoder.count(context)
        if token_count > args.budget:
            raise ValueError("Rendered context exceeds the declared tokenizer budget")
        result = {"query": doc["query"], "method": args.method, "evidence": evidence, "context": context,
                  "context_tokens": token_count, "budget_tokenizer": "all-MiniLM-L6-v2 WordPiece", "budget": args.budget,
                  "eligibility": "Caller must supply only permitted, valid memories"}
        write_json(args.output, result)
        result = {"output": str(args.output), "memories": len(evidence), "context_tokens": token_count}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
