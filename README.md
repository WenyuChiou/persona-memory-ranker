# Persona Memory Selector

This DSCI310 project tests whether an AI can give more role-consistent answers when it receives behavioral examples selected for the current situation and target trait.

## The idea in plain language

Suppose the character is careful and dependable. A client asks for an earlier deadline. A fixed persona prompt describes the character, but it does not show how the character behaves under pressure.

This project searches a bank of dialogue examples, predicts which personality trait each response expresses, and gives the most relevant examples to a local language model. We then compare the answers produced with and without those examples.

The study asks two separate questions:

1. Can course models classify the ten BIG5-CHAT generation labels?
2. Do examples chosen with those classifications help a local AI maintain the requested role?

The first question has measured results. The second still requires two human reviewers.

## Complete project flow

1. **Download one fixed dataset version.** BIG5-CHAT contains 100,000 synthetic dialogue records with five Big Five traits, each generated at a high or low setting.
2. **Clean the records.** Remove rows without an input or response, preserve the original text and source row, replace names only in model input, and record every exclusion.
3. **Keep related scenarios together.** Ten personality versions of the same situation must stay in one split. This prevents the model from seeing a near-copy of a test question during training.
4. **Turn text into numbers.** TF-IDF represents distinctive words. A fixed MiniLM encoder represents semantic similarity. Long passages are processed in overlapping windows.
5. **Train course models in R.** Multinomial logistic regression and random forest predict one of ten generation labels. Grouped cross-validation chooses the model before the test set is scored.
6. **Build one shared candidate set.** For each new question, semantic similarity retrieves the same top 50 candidates for methods B, C, and D.
7. **Compare four answer conditions.** A uses only the fixed persona prompt. B adds similar examples. C reranks them using the classifier. D adds graph relationships between situations, examples, and trait scores.
8. **Keep the comparison fair.** B, C, and D use the same candidates, at most five examples, and a conservative 2,000-byte evidence limit.
9. **Generate answers locally.** Qwen 2.5 7B is the main model and Llama 3.1 8B is the transfer check. No paid API is required.
10. **Evaluate the two outcomes separately.** Macro-F1 evaluates classification. Two blinded reviewers score role behavior, voice, relevance, fabricated history, factual errors, and invalid answers.

See the [Chinese data walkthrough](deliverables/selector/DATA_WALKTHROUGH.md) for row-level examples and the [study protocol](docs/SELECTOR_PROTOCOL.md) for the fixed experimental rules.

## Current measured results

- Raw records: 100,000
- Retained after cleaning: 99,656
- Frozen test records: 14,577
- Response-only logistic Macro-F1: 0.5878
- TF-IDF logistic Macro-F1: 0.5821
- Local answers generated and validated: 500
- Human persona-quality ratings completed: 0 of 800

The classification result does not prove that the generated answers fit the role better. That conclusion depends on the blinded human comparison.

## Demo and course materials

- [Interactive demo](https://wenyuchiou.github.io/persona-memory-ranker/selector/)
- [Measured results](reports/selector/RESULTS.md)
- [R cleaning and EDA report](reports/selector/eda.html)
- [M0 proposal](deliverables/M0-proposal.md)
- [M1 presentation](deliverables/selector/M1-presentation-v4.pptx)
- [M2 presentation](deliverables/selector/M2-presentation-v4.pptx)
- [Blind-review guide](deliverables/selector/REVIEW_GUIDE.md)

Formal rating pages and method keys stay local so the public repository cannot reveal the answer conditions to reviewers.

## Run the reusable selector

```powershell
python -m pip install -e ".[encoder]"
pms retrieve --input examples/selector_request.json --output selection.json
```

The command accepts a question, one target trait, and a list of permitted dialogue examples. It returns selected original text, source references, classification scores, and retrieval scores. It does not generate an AI answer.

## Reproduce the full study

```powershell
python -m pip install -e ".[encoder,test]"
Rscript R/selector/bootstrap.R
powershell -File scripts/run_selector.ps1 -Generate
Rscript R/selector/render.R
```

The generation stage requires local Ollama installations of `qwen2.5:7b` and `llama3.1:8b`. Raw data, embeddings, local response caches, blind-review files, and private method keys are excluded from Git.

## Data and research limits

The project uses [BIG5-CHAT](https://huggingface.co/datasets/wenkai-li/big5_chat), revision `adf1cd37997b498ff7b220827eaacdeb8aa6d905`. The official dataset card identifies the license as Apache-2.0.

The labels describe synthetic generation settings. They are not diagnoses, calibrated measurements of people, or proof of psychological causality. The project evaluates an AI role-playing method under controlled conditions.
