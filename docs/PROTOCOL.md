# Research protocol

## Question and estimand

Can classical machine learning improve retrieval of annotated source evidence for synthetic AI personas, compared with BM25, MiniLM vector search, and reciprocal rank fusion, at the same 2,000-token context budget?

The experimental unit is a query. Personas are the grouping unit for splitting, cross-validation, and uncertainty. Results concern the published synthetic benchmark and its annotations. They do not establish real-user benefit, causal psychological mechanisms, or downstream answer accuracy.

## Fixed data policy

Use text-only PersonaMem-v2, 32k histories, pinned revision from `config.py`. Preserve raw bytes. Parse mixed JSON/Python-literal CSV fields safely; never use `eval`. Preserve original query and message content. R performs canonical row cleaning and validation. Python creates source-indexed memory turns and aligns gold evidence.

The released train/val persona overlap requires an explicit repair before modeling. Preserve benchmark personas; exclude their rows from development. Pool the original train/val rows, sort personas by SHA-256 of `[310,"development",persona_id]`, and assign the first floor(80%) to training and the rest to validation. Keep original row IDs and original split metadata. No question-level random splitting.

Exclude all `system` messages from memories because they may reveal the complete synthetic profile. Retain original message indices so citations remain exact. A memory unit starts at a user message and contains its following assistant response(s). Source role labels remain visible; an assistant statement is not automatically user truth.

Align an annotation only when its role/content sequence occurs exactly once in the history. Exclude missing and ambiguous sequences with explicit reasons. No fuzzy repair. The reported retrieval population consists of aligned queries; exclusion rates must accompany every result. Long-term extensions may address currently unalignable annotations as a separately versioned experiment.

## Features and models

- BM25 uses lowercase Unicode word tokens, k1=1.5 and b=0.75.
- Frozen MiniLM uses 224-WordPiece windows with 32-token overlap. Memory/query similarity is the maximum cosine similarity over their window pairs. Windows are created without gold boundaries. Original text remains intact.
- Candidate generation unions lexical top50 and semantic top50 per query. Ties use memory ID. Each baseline and learned model ranks the same union.
- RRF uses k=60; a retriever contributes only when the memory appears in its top50.
- Five features: BM25, maximum cosine similarity, fraction of unique query words present, log(1+rendered memory token count), relative turn position.
- R `glm` binomial model standardizes features using training-only means/scales. A no-position ablation uses the first four features. Rank-deficient columns receive explicitly recorded zero coefficients after an identifiable fit.
- R `ranger` uses 300 probability trees, mtry=2, minimum node size10, seed310, and two threads. Models are compared with persona-grouped five-fold OOF predictions. Fold normalization uses fold training data only.

No gold persona, gold preference, topic label, correct/incorrect answer, gold snippet, or gold evidence-distance feature enters training predictors. Class labels are attached only after gold-blind candidate extraction. The class means annotated support versus unannotated; missing annotations can create false negatives.

## Metrics and model selection

Recall@5 divides retrieved gold units by **all** gold units for the query, including those absent from the candidate pool. MRR@10 uses the first annotated support unit. Candidate recall measures the first-stage ceiling.

Budget recall packs complete source-tagged memory units in score order. Units that do not fit are skipped; original text is not truncated to force a hit. The 2,000-token budget uses fixed MiniLM WordPiece tokens including source headers. Different production tokenizers require a final host check. A gold unit longer than the budget is an explicit budget limitation.

Report query-weighted means and 95% percentile confidence intervals from 1,000 bootstrap resamples of whole personas. Report updated/unchanged subsets and the no-position ablation. Coefficients describe retrieval associations, not causal effects.

Select the portable default using validation mean budget recall among RRF, vector, BM25, and logistic. Exact ties favor a baseline, in that order. RF remains an R comparison model. Freeze dataset/split/feature/model/research-source hashes before benchmark scoring. A development subset cannot unlock the benchmark.

## Human audit and acceptance

Generate 100 deterministic training/validation cases for source-alignment and missing-support review. A human fills reviewer, date, decision, alternative support IDs, and notes. Automated alignment checks and model-generated text never count as completed human review.

Acceptance requires reproducible metrics, source citations, no cross-persona leakage, finite scores, exact R/Python logistic agreement within 1e-8, and context-budget enforcement. A negative model result is valid. No performance gain is claimed until measured.
