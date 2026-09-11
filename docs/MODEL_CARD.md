# Model and interface card

## Intended use

Rank already-eligible candidate evidence for a persona memory application. Training uses synthetic public conversations. Evaluate domain transfer before applying the model to a real user's memory store. Retrieved assistant text remains assistant text, not certified user evidence.

## Contract

`LogisticScorer(model).score(feature_rows)` consumes exactly the frozen feature names in their declared order. Model JSON contains `schema_version`, `model_type`, `feature_version`, `feature_names`, `means`, `scales`, `coefficients`, and `intercept`. Version1 supports full and no-position logistic models. Unknown fields in feature rows, missing features, nonfinite values, incompatible schemas, and invalid scales raise an error.

The score is `sigmoid(intercept + sum(coefficient * (value - mean) / scale))`. Training labels reflect incomplete benchmark annotations. Treat scores as ranking signals rather than calibrated probabilities that a statement is true or safe to use.

The CLI takes:

```json
{
  "query": "Where can I read quietly?",
  "memories": [
    {
      "memory_id": "m-001",
      "text": "user: I like quiet libraries.",
      "source_ref": "conversation-01:message-03",
      "turn_index": 3
    }
  ]
}
```

It returns selected evidence with IDs, scores, source references and original text, plus the context block and counted budget. Duplicate IDs fail. An empty candidate list returns no evidence. The graph explorer visualizes provenance; it does not add hidden relation predictions.

## Host integration

1. The host enforces user/tenant isolation, consent, memory status, temporal validity, supersession, and any required contradictory evidence.
2. The host supplies the eligible candidate pool and versioned features, or uses the standalone lexical/semantic feature builder with its complete eligible history.
3. The scorer ranks; the host applies its mandatory-card rules and target-tokenizer budget before prompt assembly.
4. Log returned memory IDs and scorer version. Keep unknown or missing evidence explicit.

Raw BM25 features depend on the eligible corpus, and position uses observed turn order. Do not mix scores computed from a different candidate construction without a compatibility evaluation. Psychology-related metadata can be retained by a host but is not a feature in version1.

## Evaluation and limitations

See `reports/RESULTS.md` for actual validation/benchmark numbers and `artifacts/frozen_protocol.json` for the selected portable default. The RF model is a frozen R comparison artifact, not silently substituted into the Python scorer.

Known limits include synthetic data, incomplete labels, alignment exclusions, heuristic turn grouping, maximum-window matching, long units that do not fit the budget, and untested domain/language transfer. Human audit status is explicit in `reports/human_audit_status.json`.
