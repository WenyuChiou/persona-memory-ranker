"""Generate a source-backed result summary; never substitute illustrative numbers."""
from pathlib import Path
import json
import platform
from importlib.metadata import version

from persona_memory_ranker.io import read_json, read_csv, write_json
from persona_memory_ranker.pipeline import verify_evaluation, verify_frozen

root = Path(__file__).resolve().parents[1]
alignment = read_json(root/"reports/alignment.json")
manifest = read_json(root/"data/download_manifest.json")
audit_path = root/"reports/human_audit_status.json"
audit = read_json(audit_path) if audit_path.exists() else {"status": "not yet queued", "completed": 0}
lines = ["# Experimental results", "", "All numbers below come from local pipeline artifacts.", "",
         f"- Dataset revision: `{manifest['revision']}`.",
         f"- Aligned population: {alignment['aligned_queries']:,} of {alignment['input_queries']:,} queries; {alignment['excluded_queries']:,} excluded.",
         f"- Histories: {alignment['histories']:,}; memory turns: {alignment['memories']:,}; excluded system messages: {alignment['excluded_system_messages']:,}.",
         f"- Human audit: {audit['status']}; completed entries: {audit.get('completed', 0)}.",
         f"- Persona limit per split: {manifest.get('persona_limit_per_split') or 'full data'}.", ""]
for split in ("cv", "val", "benchmark"):
    path = root/f"reports/{split}_evaluation.json"
    if not path.exists():
        lines += [f"## {split}", "", "Not yet evaluated.", ""]
        continue
    result = verify_evaluation(root, split)
    if split == "benchmark":
        verify_frozen(root)
    summary = result["summary"]
    parity = "Grouped out-of-fold R predictions; full-fit predictions are excluded." if split == "cv" else f"R/Python maximum score difference: {result['r_python_max_error']:.3g}."
    lines += [f"## {split}", "", f"Evaluated queries: {result['queries']:,}. {parity}", "",
              "| Method | Recall@5 | MRR@10 | Budget recall, 95% CI | Candidate recall |",
              "|---|---:|---:|---:|---:|"]
    for method in sorted({r['method'] for r in summary}):
        m = {r['metric']: r for r in summary if r['method'] == method}
        budget = m['budget_recall']
        lines.append(f"| {method} | {m['recall_at_5']['mean']:.4f} | {m['mrr_at_10']['mean']:.4f} | {budget['mean']:.4f} [{budget['ci_low']:.4f}, {budget['ci_high']:.4f}] | {m['candidate_recall']['mean']:.4f} |")
    lines += [""]
freeze_path = root/"artifacts/frozen_protocol.json"
if freeze_path.exists():
    freeze = read_json(freeze_path)
    lines += [f"Selected portable method: **{freeze['selected_portable_method']}**, chosen using validation budget recall before benchmark scoring.", ""]
lines += ["## Interpretation limits", "", "The data and annotations are synthetic. Unannotated candidates may still support a query. Alignment failures change the evaluated population. These results do not measure downstream answer quality, real-user benefit, or psychological validity. Whole-persona bootstrap intervals account for repeated queries within a persona.", "",
          "The official train/validation CSVs overlap by persona. The project preserves the benchmark, excludes its personas from development, and repairs development splitting before training. Initial system messages containing synthetic profiles are excluded from retrieval.", ""]
(root/"reports/RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
write_json(root/"reports/runtime.json", {"python": platform.python_version(), "platform": platform.platform(),
    "data_revision": manifest['revision'],
    "python_packages": {name: version(name) for name in ("numpy", "torch", "sentence-transformers", "transformers", "tokenizers", "huggingface-hub")},
    "R": read_json(root/"artifacts/models/training_report.json")["software"]})
print(root/"reports/RESULTS.md")
