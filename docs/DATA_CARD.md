# Data card

**Source:** [PersonaMem-v2](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2), by Bowen Jiang and collaborators; [paper](https://arxiv.org/abs/2512.06688). **License:** CC BY 4.0. **Revision:** `ed956dea41521fc4499acbc63f966e0fd3c053ba`.

The authors generated personas, dialogue histories, preferences, and target questions with GPT-5. These are synthetic observations. The data include English text and multiple topics; this project uses text-only 32k histories. Demographic or health-related content is not used to estimate human traits or make consequential decisions.

## Observed release differences

| Property | Dataset card | Pinned CSV inspection |
|---|---:|---:|
| Train questions | 18,500 | 18,549 |
| Validation questions | 2,600 | 2,061 |
| Benchmark questions | 5,000 | 5,000 |
| Train/validation persona overlap | Stated absent | 735 personas |
| Development/benchmark overlap | Stated absent | Persona78; 22 development rows |

The pipeline preserves benchmark membership and repairs development splitting by persona. The complete explanation and IDs appear in `reports/split_audit.json`. Preserve this discrepancy in reports instead of copying the dataset-card counts as observed counts.

Some histories include an initial system prompt containing the expanded persona. It is excluded from retrieval. Histories have message order; do not infer calendar dates from generation filenames. Preference-update metadata is used for stratified reporting, not as an inference feature.

## Derivations

Download manifest -> CSV normalization -> R cleaning -> role-preserving turns -> exact gold alignment -> gold-blind features -> labeled candidate pairs. Source text remains unaltered, with formatting and turn grouping identified as this project's modifications.

Unalignable evidence is excluded and counted. Unannotated candidates may still be useful. The human audit remains pending until a named human completes its entries. Full synthetic persona descriptions, answer keys, and private team data do not appear in the public demo input.

## Reproducibility and storage

Raw data and large derived files stay in ignored local directories. Source URLs, revisions, SHA-256 hashes, parser decisions, exclusion reasons, and selected IDs remain in manifests/reports. Embedding caches key model revision and exact text. Tests never download a dataset or use a paid API.
