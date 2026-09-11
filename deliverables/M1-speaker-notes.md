# M1 narration draft

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

## 9. Grouped evaluation design

Model development keeps personas separated at every stage. Five-fold cross validation holds out whole personas, and each fold computes its normalization values using only its training rows. The final model uses training-only normalization as well. Validation selects the portable default before the benchmark is scored. The code records a frozen protocol containing data, model, and source hashes, which helps detect accidental changes after selection. Labels need a careful interpretation: a candidate without an annotated supporting snippet is unannotated, not proven useless. We therefore describe the supervised task as prediction of benchmark evidence labels. A separate human review is needed to assess alignment and possible missing support.

## 10. M1 findings and next steps

The first milestone delivers a reproducible preprocessing and exploratory analysis workflow. Its main finding is that the released files require stronger checks than their descriptive documentation suggests. The overlap repair and removal of profile-bearing system messages protect the experimental comparison. The alignment report makes excluded observations visible. A one-hundred-case audit queue has been designed for a human to check source alignment and possible omitted evidence. An automatically created queue does not count as a completed human review. The next stage compares the models under a fixed budget and presents the resulting evidence selections in an interactive explorer. The final research value comes from reproducible measurements, including a negative result if learned ranking does not improve the baseline.