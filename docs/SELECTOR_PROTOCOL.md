# Persona Memory Selector

## 1. Research question

With one fixed persona prompt, does selection of personality-related behavioral exemplars improve role-consistent decisions? Compare classification-aware flat retrieval with a graph using the same candidates. The study tests an Apersona-compatible component, not the entire private engine or a psychological assessment of people.

Ten labels represent five Big Five dimensions at high/low generation settings. They are mutually exclusive **dataset generation targets**, not ten kinds of people. The softmax score is not a person's probability of having a trait. Other dimensions are unobserved, not negative labels.

## 2. Literature and contribution

- [BIG5-CHAT](https://arxiv.org/abs/2410.16491): labeled synthetic behavioral examples, human-grounded generation rather than human-authored target responses.
- [PersonalityEvd](https://aclanthology.org/2024.emnlp-main.1115/): distinguishes momentary states, longer-term traits, and supporting utterances. Its original data is not redistributed here because the inspected repository did not establish a clear redistribution license.
- [InCharacter](https://github.com/Neph0s/InCharacter): personality fidelity evaluation through interviews. This project adapts the separation of character fidelity and factual recall, not its reported scores.
- [ThinkPersona](https://github.com/Hualeez/ThinkPersona): close prior work on persona graphs and grounded role-playing. A persona graph alone is not a novel contribution.
- [CAPS](https://pubmed.ncbi.nlm.nih.gov/7740090/): context-dependent behavioral patterns motivate situation-sensitive selection. Graph edges are not psychological causal claims.

Contribution: a small R-trained classifier, an auditable memory selection adapter, and controlled measurements of each component's incremental value. SEM remains a conceptual path from retrieval method through selected evidence to response behavior. No latent-variable or causal SEM is estimated from synthetic labels.

## 3. Source and cleaning

Source: [wenkai-li/big5_chat](https://huggingface.co/datasets/wenkai-li/big5_chat), Apache-2.0 according to the official dataset card.

- Revision: `adf1cd37997b498ff7b220827eaacdeb8aa6d905`.
- Raw CSV SHA-256: `1a3a81cac3d12f864c6f2926254fcfb379eb6270f6d45652cf5291dadddb206b`.
- Raw file is immutable; download is rejected on hash mismatch.
- Every retained record has an original CSV row reference, original context/response, target-speaker names, normalized model text, group, split, and memory/query role.
- Missing required fields are excluded with row-specific reasons. No generated imputations.
- Name replacement uses explicit source names. Preserve punctuation, negation and original source text.
- Transitive groups link original_index, normalized event head and anonymized nonempty input. Include incomplete rows while constructing groups to avoid breaking links by exclusion.
- Hash allocation with seed310 assigns entire groups approximately 70/15/15. Validation/test groups are independently assigned memory/query roles. Large components are reported and never split.
- `train_instruction` and labels cannot enter feature text. Scenario attributes describing PersonX are not relabeled as PersonY traits.
- R independently reads the official CSV, reconstructs the missing-text exclusions and allowed analysis columns, validates the prepared rows and split integrity, and creates EDA. Python performs the canonical source parsing, union-find grouping and fixed encoding. This division is disclosed rather than claiming every operation is in R.

## 4. Learning and retrieval

R performs grouped 3-fold CV, multinomial logistic regression, random forest, TF-IDF baseline and validation selection. Fixed MiniLM embeddings compare response-only against context-plus-response. Fitted preprocessing belongs inside each training fold. Report Macro-F1 by class and confusion rather than raw accuracy alone.

Logistic fits use a fixed 30-point decreasing regularization path for numerical warm starts, as recommended by the [glmnet documentation](https://glmnet.stanford.edu/reference/glmnet.html). Only the originally specified endpoint is scored in cross-validation. This fixes a single-lambda convergence failure observed on a real TF-IDF training fold; it does not introduce extra selected hyperparameters. Convergence warnings or a missing endpoint stop the fit. The numerical check is in `reports/selector/solver_validation.json`.

The best deployable embedding logistic model is exported independently of the best overall R comparator. RF or TF-IDF may win offline; report both `chosen_model` and `deployed_model` honestly.

Candidate retrieval uses the query-side situation embedding, top50 from the permitted memory partition. A/B/C/D use the same base prompt. B uses cosine; C combines cosine and target-label score; D adds scenario-cluster and target-trait nodes with PageRank damping0.85. K-means30 is fitted only on training situation embeddings.

Validation weights are selected from 0.25/0.5/0.75 using an explicitly weak proxy: equal weighting of known generation-label compatibility and cosine. This is not gold human relevance. Final response evaluation is separate and blinded.

All packs contain at most5 examples and have a 2,000-token ceiling. The implementation conservatively caps UTF-8 bytes at2,000 for the local byte-token models; this guarantees a smaller pack and records the bound separately from Ollama's actual prompt token count. Comparisons share this policy. The 2,000-token ceiling is not a promise to fill exactly2,000 tokens.

Graph classification edges preserve model scores; they do not imply causality. Empty memory is valid. Raw sources are cited. Behavioral examples must not be claimed as autobiographical events. Owned episodes require an explicit matching owner. Authorization/status filtering remains the caller's responsibility.

## 5. Frozen local experiment

Primary local model: qwen2.5:7b. Secondary: llama3.1:8b. No paid or cloud model calls. Record installed digest, prompt, temperature0.7, top_p0.9, seed310, output limit384 and context4096. Cache by full request plus model digest; failed/incomplete responses are never presented as valid cached answers.

Freeze model, features, data, clusters and validation weights before final test. No tuning on test classification or human ratings.

100 test groups:60 primary persona cases (six per class),20 secondary (two per class),20 factual/math controls. Each has four answers. Ten separate validation practice groups support rubric calibration. Human reviewers must rate independently before discussing disagreements. Keep answer keys private and use only validation cases in the public demo before ratings close.

Rate behavior, voice and relevance separately on1–5; flag fabricated history, factual error and invalid answers as0/1. Empty ratings stay pending. Bootstrap paired differences by scenario, averaging raters; distinguish main-model, transfer and control strata. Wide confidence intervals are inconclusive. Do not infer equivalence from non-significance.

Additional diagnostics: shuffled cluster edges, repeated seeds on20 cases, name removal and direct trait-statement masking. These are diagnostics rather than new independent people. Report unexecuted diagnostics explicitly.

Situation domains use fixed context-keyword rules for work, relationships and daily activities; multiple matches remain mixed and no matches remain other. These are transparent operational strata, not human-validated semantic labels. Their result tables are descriptive. The full prompt also receives a conservative context-capacity check including the reserved output before generation.

Generalization claims are bounded to held-out scenario groups and the two tested models. BIG5 data does not supply authentic longitudinal biographies or validate transfer to humans, private constructs, all industries, or all combined trait profiles.

## 6. Deliverables and acceptance

Keep legacy PersonaMem-v2 artifacts intact. New data live under selector subdirectories. Ship R source/notebook, Python CLI/scorer, frozen manifests, explicit tests, static demo using real cached validation answers, blinded rating material, M0 proposal and editable M1/M2 decks with scripts.

The automated study can finish while human ratings remain pending. Never publish a persona-quality gain before those ratings exist. No graph advantage means the simpler flat selector remains the integration default.
