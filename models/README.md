# Portable memory ranker

These files contain the R-trained logistic model and its exact preprocessing parameters. The default is selected on validation before scoring the isolated benchmark. No conversation corpus or model API key is needed to score already-computed numeric features.

| File | Purpose |
|---|---|
| `logistic.json` | Five-feature model: means, scales, coefficients and intercept |
| `logistic_no_position.json` | Four-feature ablation for comparison |
| `selection.json` | Validation-selected portable method and protocol identity |
| `feature_spec.json` | Encoder revision, segmentation and feature definitions |
| `provenance.json` | Frozen data, code and model hashes |
| `manifest.json` | SHA-256 values for the exported files |

Install the base package with `python -m pip install -e .`, then score compatible feature rows:

```python
from pathlib import Path
from persona_memory_ranker.retrieval import LogisticScorer

scorer = LogisticScorer.load(Path("models/logistic.json"))
scores = scorer.score(feature_rows)  # exact finite numeric fields from feature_spec.json
```

To compute features and retrieve original evidence for a new question, install the `encoder` extra and run:

```powershell
python -m persona_memory_ranker.cli retrieve --input examples/memories.json --output artifacts/retrieved.json --models models --method auto
```

Keep feature computation compatible with the pinned encoder and complete eligible history. BM25 scores and relative position depend on that history. A host memory engine supplies only permitted, valid memories and enforces any mandatory evidence rules before final context assembly.

Scores predict incomplete synthetic evidence annotations. They are ranking signals, not probabilities of truth or psychological measurements. See [the model card](../docs/MODEL_CARD.md), [data attribution](../docs/DATA_CARD.md), and [measured results](../reports/RESULTS.md). The random forest remains an R comparison model; this lightweight export uses logistic regression.
