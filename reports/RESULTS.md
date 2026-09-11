# Experimental results

All numbers below come from local pipeline artifacts.

- Dataset revision: `ed956dea41521fc4499acbc63f966e0fd3c053ba`.
- Aligned population: 22,952 of 25,584 queries; 2,632 excluded.
- Histories: 999; memory turns: 115,484; excluded system messages: 999.
- Human audit: pending human review; completed entries: 0.
- Persona limit per split: full data.

## cv

Evaluated queries: 14,822. Grouped out-of-fold R predictions; full-fit predictions are excluded.

| Method | Recall@5 | MRR@10 | Budget recall, 95% CI | Candidate recall |
|---|---:|---:|---:|---:|
| bm25 | 0.2575 | 0.2381 | 0.2889 [0.2819, 0.2958] | 0.8650 |
| logistic | 0.3578 | 0.3039 | 0.3930 [0.3848, 0.4008] | 0.8650 |
| logistic_no_position | 0.3518 | 0.2993 | 0.3857 [0.3781, 0.3936] | 0.8650 |
| random_forest | 0.3475 | 0.3054 | 0.3891 [0.3817, 0.3964] | 0.8650 |
| rrf | 0.3215 | 0.2855 | 0.3552 [0.3479, 0.3625] | 0.8650 |
| semantic | 0.3375 | 0.2888 | 0.3726 [0.3647, 0.3804] | 0.8650 |

First-stage limitation: **1,221/14,822 questions** have no annotated evidence in the candidate pool.

## val

Evaluated queries: 3,659. R/Python maximum score difference: 3.22e-15.

| Method | Recall@5 | MRR@10 | Budget recall, 95% CI | Candidate recall |
|---|---:|---:|---:|---:|
| bm25 | 0.2662 | 0.2403 | 0.2953 [0.2813, 0.3106] | 0.8576 |
| logistic | 0.3586 | 0.3103 | 0.3970 [0.3817, 0.4136] | 0.8576 |
| logistic_no_position | 0.3544 | 0.3067 | 0.3911 [0.3759, 0.4084] | 0.8576 |
| random_forest | 0.3526 | 0.3069 | 0.3883 [0.3736, 0.4049] | 0.8576 |
| rrf | 0.3247 | 0.2902 | 0.3585 [0.3439, 0.3749] | 0.8576 |
| semantic | 0.3391 | 0.2950 | 0.3740 [0.3592, 0.3914] | 0.8576 |

First-stage limitation: **335/3,659 questions** have no annotated evidence in the candidate pool.

Exploratory paired comparisons use the same 1,000 whole-persona resamples. Positive differences favor logistic. Intervals are unadjusted 95% percentile intervals; comparisons do not change the frozen method selection.

| Paired comparison | Budget recall difference (percentage points), 95% CI |
|---|---:|
| logistic minus logistic_no_position | +0.59 [-0.17, +1.36] |
| logistic minus rrf | +3.85 [+2.84, +4.86] |
| logistic minus semantic | +2.30 [+1.52, +3.14] |

Example logistic losses against RRF, selected **after evaluation** by largest recall loss, then query ID. These examples are not a representative sample and do not affect the independently selected demo cases.

| Source query ID | Logistic budget recall | RRF budget recall | Update label |
|---|---:|---:|---|
| train-00075 | 0.000 | 1.000 | FALSE |
| train-00241 | 0.000 | 1.000 | FALSE |
| train-00598 | 0.000 | 1.000 | FALSE |

Query IDs index the processed query files, which retain original split/row IDs and source-memory references.

| Update annotation | Method | Budget recall | Questions |
|---|---|---:|---:|
| true | bm25 | 0.6771 | 830 |
| true | logistic | 0.9259 | 830 |
| true | logistic_no_position | 0.9343 | 830 |
| true | random_forest | 0.8301 | 830 |
| true | rrf | 0.8169 | 830 |
| true | semantic | 0.9259 | 830 |
| false | bm25 | 0.1832 | 2829 |
| false | logistic | 0.2418 | 2829 |
| false | logistic_no_position | 0.2318 | 2829 |
| false | random_forest | 0.2586 | 2829 |
| false | rrf | 0.2240 | 2829 |
| false | semantic | 0.2120 | 2829 |

## benchmark

Evaluated queries: 4,471. R/Python maximum score difference: 3.44e-15.

| Method | Recall@5 | MRR@10 | Budget recall, 95% CI | Candidate recall |
|---|---:|---:|---:|---:|
| bm25 | 0.2559 | 0.2475 | 0.2900 [0.2756, 0.3031] | 0.8697 |
| logistic | 0.3583 | 0.3110 | 0.3941 [0.3782, 0.4093] | 0.8697 |
| logistic_no_position | 0.3548 | 0.3068 | 0.3916 [0.3756, 0.4069] | 0.8697 |
| random_forest | 0.3416 | 0.3050 | 0.3816 [0.3659, 0.3958] | 0.8697 |
| rrf | 0.3202 | 0.2953 | 0.3548 [0.3407, 0.3695] | 0.8697 |
| semantic | 0.3449 | 0.2981 | 0.3804 [0.3648, 0.3952] | 0.8697 |

First-stage limitation: **358/4,471 questions** have no annotated evidence in the candidate pool.

Exploratory paired comparisons use the same 1,000 whole-persona resamples. Positive differences favor logistic. Intervals are unadjusted 95% percentile intervals; comparisons do not change the frozen method selection.

| Paired comparison | Budget recall difference (percentage points), 95% CI |
|---|---:|
| logistic minus logistic_no_position | +0.25 [-0.32, +0.76] |
| logistic minus rrf | +3.93 [+3.05, +4.71] |
| logistic minus semantic | +1.37 [+0.75, +1.96] |

Example logistic losses against RRF, selected **after evaluation** by largest recall loss, then query ID. These examples are not a representative sample and do not affect the independently selected demo cases.

| Source query ID | Logistic budget recall | RRF budget recall | Update label |
|---|---:|---:|---|
| benchmark-00024 | 0.000 | 1.000 | FALSE |
| benchmark-00072 | 0.000 | 1.000 | FALSE |
| benchmark-00188 | 0.000 | 1.000 | FALSE |

Query IDs index the processed query files, which retain original split/row IDs and source-memory references.

| Update annotation | Method | Budget recall | Questions |
|---|---|---:|---:|
| true | bm25 | 0.6649 | 1031 |
| true | logistic | 0.9263 | 1031 |
| true | logistic_no_position | 0.9384 | 1031 |
| true | random_forest | 0.8274 | 1031 |
| true | rrf | 0.8225 | 1031 |
| true | semantic | 0.9374 | 1031 |
| false | bm25 | 0.1776 | 3440 |
| false | logistic | 0.2346 | 3440 |
| false | logistic_no_position | 0.2277 | 3440 |
| false | random_forest | 0.2480 | 3440 |
| false | rrf | 0.2146 | 3440 |
| false | semantic | 0.2135 | 3440 |

Selected portable method: **logistic**, chosen using validation budget recall before benchmark scoring.

## Interpretation limits

The data and annotations are synthetic. Unannotated candidates may still support a query. Alignment failures change the evaluated population. These results do not measure downstream answer quality, real-user benefit, or psychological validity. Whole-persona bootstrap intervals account for repeated queries within a persona.

The official train/validation CSVs overlap by persona. The project preserves the benchmark, excludes its personas from development, and repairs development splitting before training. Initial system messages containing synthetic profiles are excluded from retrieval.
