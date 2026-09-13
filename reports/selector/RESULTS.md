# Persona Memory Selector results

These results concern synthetic generation-target classification. Human persona-quality effects remain pending until actual blinded ratings are complete.

## Data

Raw: 100000; retained: 99656; excluded: 344.

## Validation classification

| Model | Macro-F1 |
|---|---:|
| context_glmnet | 0.3969 |
| context_rf | 0.2697 |
| response_glmnet | 0.5829 |
| response_rf | 0.4452 |
| tfidf_glmnet | 0.5802 |

Selected offline model: response_glmnet. Portable model: response_glmnet.

## Frozen test classification

| Model | Test Macro-F1 | Test rows |
|---|---:|---:|
| context_glmnet | 0.3992 | 14577 |
| context_rf | 0.2610 | 14577 |
| response_glmnet | 0.5878 | 14577 |
| response_rf | 0.4460 | 14577 |
| tfidf_glmnet | 0.5821 | 14577 |

Response-only logistic and TF-IDF logistic are close on this split; no significance claim is made. Adding context to the fixed classification representation reduced performance. Context is still used for retrieval.

## Local generation

```json
{
  "val": {
    "groups": 10,
    "answers": 40,
    "valid_completions": 40,
    "invalid_completions": 0
  },
  "test": {
    "groups": 100,
    "answers": 400,
    "valid_completions": 400,
    "invalid_completions": 0
  },
  "test-diagnostics": {
    "groups": 20,
    "answers": 60,
    "valid_completions": 60,
    "invalid_completions": 0
  }
}
```

## Human evaluation

Status: pending; completed rating rows: 0. Two reviewers x100 situations x4 answers =800 expected rows.

No ranking method is declared better at persona behavior from classifier scores alone. Graph advantage is a separate D-minus-C hypothesis.

## Reproducibility

See [the artifact manifest](artifact_manifest.json), [the frozen protocol copy](frozen.json), source_manifest.json, features.json and the local generation cache. Measured data, model and result copies retain their original bytes and hashes; frozen.json records the documented package-only encoder move. Packs use a conservative 2,000-byte upper bound and at most 5 examples. Local AI completion counts are not human quality scores.

The 200-group name-shift diagnostic is reported in stress.json. Direct-self-description masking changed zero selected rows, so that diagnostic provides no evidence about removal of explicit trait statements.
