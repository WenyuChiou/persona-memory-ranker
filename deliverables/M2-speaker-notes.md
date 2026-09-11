# M2 narration draft

Synthetic narration is a presentation aid; review the content before course use.

## 1. Persona Memory Ranker

This project studies a practical part of personalized artificial intelligence: choosing which memories belong in the context for the next answer. A system may store a long conversation history and still retrieve the wrong evidence. The goal here is to compare several retrieval methods under the same constraints, then expose the results through a reusable interface. The course methods include logistic regression, tree ensembles, grouped cross validation, and exploratory data analysis in R. The data are synthetic conversations from PersonaMem version two. The claims in this presentation concern retrieval of annotated evidence. They do not establish psychological validity or benefit for real users.

## 2. Research question

The research question is whether classical machine learning improves the selection of annotated evidence compared with lexical search, vector search, and a hybrid of both. A fair comparison requires holding the candidate pool and context budget constant. Otherwise a method could look stronger simply because it sees more material or returns a longer prompt. Each result keeps a memory identifier and a source reference. This allows us to inspect why the method selected a passage and whether that passage supports the query. The application is a memory engine for a persona agent, but the experiment evaluates retrieval directly. Response generation and personality inference are separate questions.

## 3. Public dataset and observed counts

The source is a publicly available dataset with a Creative Commons attribution license. We fix a particular release so that another person can reproduce the same input bytes. The dataset contains generated personas, histories, queries, and annotated conversation snippets. Downloading the actual files revealed a discrepancy: the training CSV has eighteen thousand five hundred forty-nine questions, validation has two thousand sixty-one, and the benchmark has five thousand. These observed counts differ from the rounded descriptions in the dataset card. We therefore use computed manifests and reports for the analysis, rather than treating the card as a substitute for inspecting the data. The generated nature of every example remains an important limitation.

## 4. Persona overlap in the released splits

The most consequential quality finding is overlap between the released partitions. All seven hundred thirty-five validation personas also appear in training. In addition, persona seventy-eight appears in the development data and in the benchmark. Splitting individual questions would therefore expose the model to the same people in training and validation. The repair preserves the two hundred benchmark personas. Their development rows are excluded. We then pool the remaining development rows and divide the seven hundred ninety-nine personas with a fixed hash and an eighty-twenty rule. This yields six hundred thirty-nine training personas and one hundred sixty validation personas. Original source row identifiers are retained for traceability.

## 5. Cleaning with preserved source text

The histories themselves also need inspection. Their initial system messages contain expanded synthetic profiles. Those profiles were generation context; indexing them would give the retrieval system privileged information. We exclude system messages while retaining original source indices. One assistant record contains only a role and no content. It is recorded as a missing message, rather than being filled with invented text. Another practical issue was Unicode handling between Python and R on Windows. We corrected the process locale and tested accented text, Chinese characters, emoji, quotes, and multiline fields. These checks matter because even an invisible text conversion can prevent exact evidence alignment or alter the source that appears in the demo.

## 6. Evidence alignment and exclusions

The annotation names a supporting conversation snippet. We locate that snippet in the original history by matching the complete role and content sequence. The match must be unique. We do not repair a missing snippet with fuzzy matching or let a language model invent the location. In this run, 22952 queries aligned out of 25584 cleaned queries. The remaining 2632 queries are listed in the alignment report. This exclusion changes the population on which retrieval metrics are computed, so the rate must accompany the result. A high score on the aligned subset cannot be described as performance on every original question. Inspecting the excluded cases is part of the data-quality analysis.

## 7. Memory units and source references

The retrieval unit begins with a user turn and includes its following assistant response. This is a simple and reproducible segmentation rule. It does not use the gold annotation to decide where a unit starts. Long units are encoded through overlapping windows because the selected MiniLM encoder otherwise truncates long inputs. Similarity uses the strongest window match, while the returned evidence retains the complete original unit. Role labels remain visible. In particular, an assistant statement is not automatically evidence about the user's true preference. The memory identifier and original message indices make the unit traceable. Segmentation choices can influence results and should be revisited in a separately versioned experiment.

## 8. Exploratory analysis

Exploratory analysis describes both the retained and excluded populations. The R notebook reports the number of queries by partition, history and memory sizes, evidence positions, query topics, and preference-update labels. Position deserves special care. The history provides message order, but it does not provide an event timestamp for every conversation. A date embedded in a generated filename is not evidence of when a user's preference was valid. We therefore use relative conversational position as a feature and describe it that way. Topic and update labels help stratify the evaluation, but they are not given to the ranker as oracle features. The notebook connects these descriptive checks to the later modeling decisions.

## 9. Candidate generation and models

All methods rank the same union of lexical and semantic candidates. The lexical method uses BM25, the semantic method uses a frozen sentence encoder, and the hybrid uses reciprocal rank fusion. The course models are logistic regression and a random forest with three hundred trees. The predictor set contains lexical relevance, semantic similarity, word overlap, memory length, and relative conversational position. We also fit a logistic model without position to examine how much the apparent benefit depends on recency information. The encoder is a fixed feature extractor, not a model trained for this assignment. This distinction keeps the contribution of the course methods visible and the experiment computationally feasible on a CPU.

## 10. Fixed-budget retrieval

A ranking is useful only if the evidence fits into the available context. The experiment therefore packs complete memory units in score order until the fixed two-thousand-token budget is exhausted. Source headers count toward the cost. A unit that does not fit is skipped, and its text is not shortened just to force a successful hit. Budget recall measures how much annotated evidence survives this process. Recall at five and reciprocal rank at ten provide complementary views of ranking quality. Candidate recall measures whether the supporting units were retrieved at all. Every recall denominator includes all annotated units for a query, including those absent from the candidate pool, so first-stage failures cannot disappear from the evaluation.

## 11. Benchmark evidence recall

The benchmark comparison now uses the frozen methods and the isolated benchmark personas. It includes 4471 aligned questions. Budget recall is the main practical comparison because the application must operate within a context limit. The observed mean results are: BM25 29.0%; Vector 38.0%; Hybrid 35.5%; Logistic 39.4%; No position 39.2%; Forest 38.2%. These are measurements from the generated report, not illustrative numbers. Confidence intervals are computed by resampling entire personas one thousand times. This keeps repeated questions from the same persona together during uncertainty estimation. The intervals describe variation in this synthetic sample. They do not account for every possible deployment shift, and they should not be interpreted as evidence that a method understands human psychology.

## 12. Ranking performance and position

Recall at five shows whether supporting evidence appears near the top of the ranked list before budget packing. Comparing this measure with budget recall helps separate relevance from length effects. A model may rank a long correct unit highly but still fail to include it under the budget. The position ablation tests whether conversational order contributes useful information beyond lexical and semantic relevance. The full logistic model differs from the no-position model by 0.25 percentage points of budget recall, with a paired ninety-five percent interval from -0.32 to +0.76 percentage points. The interval includes zero, so a stable contribution from position is not established. The fitted position coefficient is -0.257: later source order reduces the score when the other features are held fixed. This does not establish a universal rule about which memories an agent should keep.

## 13. Where the ranker still fails

The average result hides important differences between query groups. On questions carrying the dataset's update annotation, logistic budget recall is 92.63%, compared with 93.74% for vector retrieval. On other questions, the corresponding results are 23.46% and 21.35%. Thus the learned ranker does not outperform vector search in every group. These are descriptive subgroup results, and the update label is not an inference-time feature. We do not change the default or introduce a benchmark-tuned routing rule. Another limit occurs before ranking: 358 questions have no annotated supporting memory in the candidate pool. A reranker cannot recover evidence that was never retrieved. These failures identify separate targets for a future experiment.

## 14. Interactive evidence explorer

The interactive explorer makes the experiment inspectable. A visitor selects a query, changes the ranking method, and sees which memories fit into the chosen context budget. The table distinguishes annotated supporting evidence from unannotated candidates. Clicking a memory opens the original role-labelled messages and a source reference. The provenance graph connects the question, selected memory units, and their history source. It is a visualization of evidence flow, not a psychological model. The public page replays saved experiment cases without running a language model in the browser. A separate local command-line interface accepts a new query and a permitted memory collection, computes features, and returns a source-preserving context block.

## 15. Reusable memory-engine interface

The portable default selected on validation is logistic. The Python interface also supports the exported logistic scorer. R writes the feature order, training means, scales, coefficients, and intercept into a versioned JSON artifact. Python applies the same transformation. The maximum observed discrepancy on benchmark scores is 3.44e-15, which is below the specified numerical tolerance. This interface belongs inside a larger memory system, after the host has decided which memories are permitted and currently valid. The scorer does not grant consent, resolve contradictions, or promote inferred memory records to accepted facts. The host also checks its own language model tokenizer before assembling the final prompt, because this benchmark uses a fixed WordPiece budget.

## 16. Conclusions and remaining limits

The completed system provides a reproducible path from public raw conversations to R models, retrieval metrics, and a reusable evidence interface. The frozen logistic model differs from vector retrieval by 1.37 percentage points of benchmark budget recall. That is a measured retrieval result, not a downstream answer-quality result. The release audit also matters: persona overlap and profile-bearing system prompts could otherwise produce misleading results. The data are synthetic, annotations may omit useful memories, and exact alignment excludes a portion of the questions. The human audit queue still requires actual reviewer decisions. Integrating the scorer into a real memory engine requires a separate domain-transfer check. The practical outcome is a tested module and an evaluation framework that can support that next experiment without rebuilding the course project.