"""Generate a source-backed result summary; never substitute illustrative numbers."""
from pathlib import Path
import json
import platform
from importlib.metadata import version

from persona_memory_ranker.io import read_json, read_csv, write_json
from persona_memory_ranker.pipeline import verify_evaluation, verify_frozen
from persona_memory_ranker.metrics import aggregate

root = Path(__file__).resolve().parents[1]
alignment = read_json(root/"reports/alignment.json")
manifest = read_json(root/"data/download_manifest.json")
audit_path = root/"reports/human_audit_status.json"
audit = read_json(audit_path) if audit_path.exists() else {"status": "not yet queued", "completed": 0}
paired_results = {}
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
    query_rows = read_csv(root/f"reports/{split}_query_metrics.csv")
    by_query = {}
    for row in query_rows:
        by_query.setdefault(row['query_id'], {})[row['method']] = row
    missing = sum(float(methods['rrf']['candidate_recall']) == 0 for methods in by_query.values())
    lines += [f"First-stage limitation: **{missing:,}/{len(by_query):,} questions** have no annotated evidence in the candidate pool.", ""]
    if split != "cv":
        paired_rows = []
        for qid, methods in by_query.items():
            learned = methods['logistic']
            for comparison in ('semantic', 'rrf', 'logistic_no_position'):
                baseline = methods[comparison]
                if learned['persona_id'] != baseline['persona_id']:
                    raise ValueError("Paired metrics disagree on persona identity")
                paired_rows.append({'query_id': qid, 'persona_id': learned['persona_id'],
                    'method': f'logistic minus {comparison}',
                    **{key: float(learned[key])-float(baseline[key]) for key in
                       ('recall_at_5', 'mrr_at_10', 'budget_recall', 'candidate_recall')}})
        differences_summary = aggregate(paired_rows)
        paired_results[split] = {'source_metrics_sha256': result['output_sha256'][f'reports/{split}_query_metrics.csv'],
                                 'summary': differences_summary}
        lines += ["Exploratory paired comparisons use the same 1,000 whole-persona resamples. Positive differences favor logistic. Intervals are unadjusted 95% percentile intervals; comparisons do not change the frozen method selection.", "",
                  "| Paired comparison | Budget recall difference (percentage points), 95% CI |",
                  "|---|---:|"]
        for item in differences_summary:
            if item['metric'] == 'budget_recall':
                lines.append(f"| {item['method']} | {100*item['mean']:+.2f} [{100*item['ci_low']:+.2f}, {100*item['ci_high']:+.2f}] |")
        lines += [""]
        differences = [(float(methods['logistic']['budget_recall'])-float(methods['rrf']['budget_recall']), qid,
                        methods['logistic'], methods['rrf']) for qid, methods in by_query.items()]
        losses = sorted((r for r in differences if r[0] < 0), key=lambda r: (r[0], r[1]))[:3]
        if losses:
            lines += ["Example logistic losses against RRF, selected **after evaluation** by largest recall loss, then query ID. These examples are not a representative sample and do not affect the independently selected demo cases.", "",
                      "| Source query ID | Logistic budget recall | RRF budget recall | Update label |",
                      "|---|---:|---:|---|"]
            for _, qid, learned, baseline in losses:
                lines.append(f"| {qid} | {float(learned['budget_recall']):.3f} | {float(baseline['budget_recall']):.3f} | {learned['updated']} |")
            lines += ["", "Query IDs index the processed query files, which retain original split/row IDs and source-memory references.", ""]
        updates = root/f"reports/{split}_updates.csv"
        if updates.exists():
            lines += ["| Update annotation | Method | Budget recall | Questions |", "|---|---|---:|---:|"]
            for row in read_csv(updates):
                if row['metric'] == 'budget_recall':
                    lines.append(f"| {row['updated']} | {row['method']} | {float(row['mean']):.4f} | {row['queries']} |")
            lines += [""]
freeze_path = root/"artifacts/frozen_protocol.json"
if freeze_path.exists():
    freeze = read_json(freeze_path)
    lines += [f"Selected portable method: **{freeze['selected_portable_method']}**, chosen using validation budget recall before benchmark scoring.", ""]
lines += ["## Interpretation limits", "", "The data and annotations are synthetic. Unannotated candidates may still support a query. Alignment failures change the evaluated population. These results do not measure downstream answer quality, real-user benefit, or psychological validity. Whole-persona bootstrap intervals account for repeated queries within a persona.", "",
          "The official train/validation CSVs overlap by persona. The project preserves the benchmark, excludes its personas from development, and repairs development splitting before training. Initial system messages containing synthetic profiles are excluded from retrieval.", ""]
(root/"reports/RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
write_json(root/"reports/paired_comparisons.json", paired_results)
write_json(root/"reports/runtime.json", {"python": platform.python_version(), "platform": platform.platform(),
    "data_revision": manifest['revision'],
    "python_packages": {name: version(name) for name in ("numpy", "torch", "sentence-transformers", "transformers", "tokenizers", "huggingface-hub")},
    "R": read_json(root/"artifacts/models/training_report.json")["software"]})
print(root/"reports/RESULTS.md")
