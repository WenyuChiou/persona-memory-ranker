# Persona Memory Ranker

**Which memories should an AI persona retrieve for its next answer?**

This DSCI310 project compares lexical, semantic, hybrid, and R-trained evidence rankers under a fixed context budget. It preserves source text and evaluates retrieval on synthetic persona conversations. The reusable scorer belongs after a host application's permission, time-validity, and memory-status checks.

The project studies retrieval quality. It does not diagnose personality or establish psychological causality.

- [Interactive evidence explorer](web/index.html)
- [Research protocol](docs/PROTOCOL.md)
- [M0 proposal](deliverables/M0-proposal.md)
- [Generated results](reports/RESULTS.md)
- [Data card](docs/DATA_CARD.md) and [model/interface card](docs/MODEL_CARD.md)

## A consequential data-cleaning finding

The pinned PersonaMem-v2 release differs from its dataset card. The actual CSVs contain **18,549 training**, **2,061 validation**, and **5,000 benchmark** questions. All 735 validation personas overlap the official training personas; persona `78` also overlaps the benchmark. Some histories include a `system` message containing the complete synthetic persona.

We preserve the 200 benchmark personas, remove their development rows, and split the remaining 799 development personas by a fixed hash into **639 train / 160 validation**. Original split and row IDs remain traceable. Complete-persona system messages never enter the memory index. See `reports/split_audit.json` and `reports/alignment.json` for computed evidence.

## Reproduce

Python 3.11+ and R are required. The executed environment uses Python 3.14 and R 4.6.1 on Windows CPU. A GPU and paid LLM API are unnecessary.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[encoder,test]"
Rscript R/bootstrap.R
python -m pytest -q
Rscript tests/test_r_pipeline.R
```

If `Rscript` is not on PATH, use its installed absolute path. The Python CLI also discovers standard Windows R installations. R packages install into the project-local `.Rlib` directory.

Linux source installations of the notebook dependencies require the libuv development headers (`sudo apt-get install libuv1-dev` on Ubuntu). Pandoc is also required to render HTML; `R/bootstrap.R` reports whether it is available.

Start with the labelled development smoke run:

```powershell
powershell -File scripts/run_experiment.ps1 -Smoke
```

Run the full experiment in a fresh checkout or before freezing a protocol:

```powershell
powershell -File scripts/run_experiment.ps1
Rscript R/render.R --output reports/eda.html
Rscript R/plot_results.R --split benchmark --output reports/benchmark-budget-recall.png
python scripts/build_report.py
python scripts/export_models.py
python scripts/build_demo.py --split benchmark
```

The script downloads pinned source files, calls R cleaning, aligns evidence, builds development features, trains R models, checks R/Python parity, evaluates validation, freezes the protocol, and finally evaluates the benchmark. A subset cannot unlock the benchmark. Freeze manifests reject changes to data, models, or research code. Downloads and embedding caches are reused by content identity.

Grouped out-of-fold predictions are evaluated separately as `cv`, using the full training-query gold denominators. Reports, notebooks, slides and the demo verify saved evaluation inputs and metric-table hashes before displaying results. Regenerating an audit queue refuses to overwrite any entered reviewer fields.

Large data, embeddings, R libraries, full R training objects, and videos stay out of Git. Small portable JSON models and feature definitions are exported to `models/`. Download manifests contain source URLs, SHA-256 hashes, sizes, and selected persona IDs. The 100-case human audit queue is generated in `reports/private/human_audit_100.csv`; an empty reviewer field is **pending**, never a completed review.

## Try the reusable interface

```powershell
python -m persona_memory_ranker.cli retrieve --input examples/memories.json --output artifacts/example-result.json --method rrf
python -m persona_memory_ranker.cli retrieve --input examples/memories.json --output artifacts/example-logistic.json --method logistic --models models
```

Input is `{"query": "...", "memories": [...]}`. Each memory supplies `memory_id`, `text`, `source_ref`, and nonnegative integer `turn_index`. The output includes ranked original evidence, scores, a context block, and its token count. The CLI uses the frozen MiniLM WordPiece tokenizer for budget accounting; a production host must also enforce its target model's tokenizer budget.

The scorer accepts only the frozen numeric feature allowlist. It cannot accept gold answers, infer authorization, promote candidate memories to accepted status, or rewrite source evidence. [Interface details](docs/MODEL_CARD.md).

For scorer-only use, the base package requires NumPy; the optional `encoder` extra supplies text encoding. CLI `--method auto` uses a supplied `selection.json` when available, otherwise the explicit untrained hybrid baseline.

## Course deliverables

| Milestone | Due in 2026 (Eastern) | Artifact |
|---|---|---|
| M0 | September 12, 11:59 PM | Title, abstract, data source; Chrome form draft |
| M1 | October 10, 11:59 PM | R preprocessing/EDA notebook; 10-minute presentation |
| M2 | November 28, 11:59 PM | Source, experimental results, interactive demo, PowerPoint; 15-minute presentation |

The notebook and presentations must disclose unresolved human annotation review and the limits of synthetic data. Review the narrative before using it as a personal course presentation.

## Sources and licenses

- [PersonaMem-v2 dataset](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2), revision `ed956dea41521fc4499acbc63f966e0fd3c053ba`, **CC BY 4.0**. Conversation excerpts retain this attribution; the project changes their formatting and turn grouping.
- [Dataset authors' repository](https://github.com/bowen-upenn/PersonaMem-v2) and [paper](https://arxiv.org/abs/2512.06688).
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, **Apache 2.0**. The encoder is frozen, not trained for this course project.
- Original project code: MIT. No private team corpus, internal engine code, or Slack material is included.
