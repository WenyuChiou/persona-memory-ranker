# Persona Memory Selector model and interface card

## Purpose

Predict the ten BIG5-CHAT generation targets from context and a behavioral response, then rank permitted examples for an existing persona prompt. This is an experimental selector for role-playing AI. It is not a personality assessment of real people.

The classifier is trained in R. `models/selector/logistic.json` stores the exact class order, input-feature order, center/scale, selected feature indices, intercepts and coefficient matrix. Python evaluates the same multinomial softmax. The actual export parity result is recorded in `reports/selector/acceptance.json`.

## Reusable assets

- `models/selector/logistic.json`: portable classifier, independent of R at inference.
- `artifacts/selector/clusters.json`: 30 training-fitted situation centers.
- `artifacts/selector/weights.json`: validation-selected flat and graph weights.
- `examples/selector_request.json`: five public training examples from different groups, with a new illustrative query. No query reference answer or annotation labels are supplied to the selector.

The two small files under `artifacts/selector` are versioned deployment assets; other training matrices, caches, response keys and R bundles remain local.

## Local use

```powershell
python -m pip install -e '.[encoder]'
pms retrieve --input examples/selector_request.json --output artifacts/selector/example-selection.json
```

This computes retrieval scores and a memory block. It does not call a paid API or generate an answer. The fixed sentence encoder may be downloaded on first use. R and Ollama are needed for the full study, not for this portable retrieval command.

The request accepts a nonempty `query`, one target `trait`, a list of permitted `memories`, optional `persona_id`, optional `method` (default C), and integer `budget` from 0 to 2,000. Each memory includes ID, context, response, source and memory type. `owned_episode` additionally requires an owner that matches the request persona. The host application must enforce permissions, timestamps and memory validity before calling the selector.

Outputs retain the supplied text, source references, classification scores, retrieval scores and memory types. They do not insert a label into the source text or manufacture an episode. The memory budget counts full rendered memory text with a conservative UTF-8 byte upper bound, at most five examples. The caller must also budget its system prompt, current query, other context and generated answer.

To reproduce the course experiment, place the returned example block beside the fixed persona prompt:

```python
from pathlib import Path
from persona_memory_selector.interface import retrieve

selection = retrieve(Path(project_directory), {
    "query": current_question,
    "trait": "conscientiousness_high",
    "persona_id": active_persona_id,
    "memories": permitted_memories,  # Already checked by the host application.
    "method": "C",
    "budget": 2000,
})
messages = [{"role": "system", "content": existing_persona_prompt}]
if selection["memory_block"]:
    messages.append({"role": "system", "content": selection["memory_block"]})
messages.append({"role": "user", "content": current_question})
```

The variables in this snippet are placeholders for the experiment runner; the selector itself does not generate an answer. Classification can be cached when memory text and model versions stay unchanged. The study retrieves examples under a supplied trait target; it does not learn or retrieve the complete system prompt.

## Evaluation boundaries

The best offline classifier can differ from the exported logistic classifier. Report both identities from `artifacts/selector/selection.json`; do not substitute logistic's score for a winning random forest or TF-IDF model.

Ten softmax outputs are mutually exclusive generation targets in this training task. They are not ten kinds of people, calibrated probabilities of human traits, or a complete five-dimensional personality profile. A new construct vocabulary or multi-trait persona needs separate labels and validation.

The fixed encoder is English-oriented. Name masking is used during training; the external adapter preserves caller text, so name-shift diagnostics are relevant. Chinese effectiveness, authentic longitudinal memory and transfer to arbitrary industries have not been established.

Human response ratings remain pending until both reviewers complete the provided files. A high classification score alone does not establish role consistency, and a graph benefit must be measured against flat ranking. Keep C as the simple experimental default while this evidence is pending.

## Attribution

Training examples come from [BIG5-CHAT](https://huggingface.co/datasets/wenkai-li/big5_chat), dataset revision `adf1cd37997b498ff7b220827eaacdeb8aa6d905`, whose official card identifies Apache-2.0. The fixed encoder is [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2); its exact revision and pooling policy are recorded in `reports/selector/features.json`. The repository contains public course-project data and generated artifacts only.

## Current measured model

The selected and exported model is response-only logistic regression: validation Macro-F1 0.5829, frozen-test Macro-F1 0.5878 on 14,577 records. The TF-IDF baseline scores 0.5821 on the same test. These are generation-target classification scores, with no demonstrated significance between those two models. See `reports/selector/selection.json` and `test_metrics.csv`.

The five-memory CLI request is an interface fixture. It checks valid selection and citations; it is not evidence of improved persona behavior. The new-memory caller is responsible for identifying the target speaker in `response` and supplying only authorized memories.
