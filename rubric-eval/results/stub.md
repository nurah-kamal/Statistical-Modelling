# surface-counts

30 essays × 6 items = 180 calls. 0 failed after a retry. 0 reasons ran past one sentence.

## Against the human score, on the 1-10 scale

The human score is the mean of two raters, doubled.

| Item | Exact | Within 1 | Mean gap | Bias | r | QWK |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cohesion | 13% | 33% | 2.10 | +1.50 | -0.08 | -0.05 |
| syntax | 17% | 40% | 2.47 | +1.80 | -0.28 | -0.16 |
| vocabulary | 40% | 73% | 0.93 | -0.27 | 0.42 | 0.41 |
| phraseology | 13% | 50% | 1.87 | -1.87 | 0.24 | 0.07 |
| grammar | 10% | 50% | 1.83 | +0.43 | 0.46 | 0.41 |
| conventions | 10% | 37% | 2.20 | +0.93 | 0.58 | 0.44 |
| **all** | **17%** | **47%** | **1.90** | **+0.42** | **0.19** | **0.17** |

Bias above zero means the model scores higher than the people. r is the correlation, which ignores a constant offset; QWK is quadratic weighted kappa, which does not.

## How the scores spread

| Score | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Model | 7 | 3 | 5 | 21 | 22 | 28 | 28 | 37 | 18 | 11 |
| Human | 0 | 1 | 4 | 23 | 42 | 48 | 28 | 28 | 6 | 0 |

## With the average gap taken off

Adding 0 to every score (capped at 10): exact 17%, within 1 47%, mean gap 1.90, QWK 0.17. The shift is worked out on these same essays, so this is the best case, not a result.

## Reasons that mention spelling, capitals or punctuation

| Item | Reasons |
| --- | ---: |
| cohesion | 0 of 30 |
| syntax | 0 of 30 |
| vocabulary | 0 of 30 |
| phraseology | 0 of 30 |
| grammar | 0 of 30 |
| conventions | 30 of 30 |

## Against each rater, on the raters' own 1-5 scale

The model's 1-10 score is put back into its band (1-2 is level 1, 9-10 is level 5) and compared with each rater on their own. The last row is the two raters compared with each other, which is the bar to clear.

| Comparison | Exact | Within 1 | r | QWK |
| --- | ---: | ---: | ---: | ---: |
| Model vs rater 1 | 33% | 72% | 0.20 | 0.18 |
| Model vs rater 2 | 34% | 77% | 0.18 | 0.15 |
| Rater 1 vs rater 2 | 54% | 97% | 0.59 | 0.59 |

| Item | Model vs a rater, exact | Raters vs each other, exact | Model QWK | Raters QWK |
| --- | ---: | ---: | ---: | ---: |
| cohesion | 22% | 30% | 0.02 | 0.38 |
| syntax | 30% | 53% | -0.13 | 0.54 |
| vocabulary | 63% | 67% | 0.41 | 0.66 |
| phraseology | 43% | 53% | 0.09 | 0.58 |
| grammar | 32% | 57% | 0.36 | 0.70 |
| conventions | 12% | 67% | 0.37 | 0.68 |

## Where the model and the people disagree most

| Essay | Item | Model | Human | Gap | The model's reason |
| --- | --- | ---: | ---: | ---: | --- |
| CA3EB3AA11FC | cohesion | 10 | 3 | +7 | 18 paragraphs and 1 linking words. |
| 6B8C02811FF6 | syntax | 10 | 4 | +6 | 40.9 words per sentence. |
| 171E655CC8F1 | syntax | 10 | 4 | +6 | 51.5 words per sentence. |
| 2B621C2F0716 | cohesion | 10 | 5 | +5 | 8 paragraphs and 8 linking words. |
| 6B8C02811FF6 | cohesion | 8 | 3 | +5 | 6 paragraphs and 7 linking words. |
| C65705E1E072 | syntax | 9 | 4 | +5 | 26.1 words per sentence. |
| BA89E5BB44E3 | conventions | 9 | 4 | +5 | 0.0 spelling, apostrophe and capital slips per 100 words. |
| 605C1B4CE788 | syntax | 10 | 5 | +5 | 47.0 words per sentence. |
