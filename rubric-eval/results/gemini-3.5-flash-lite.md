# gemini-3.5-flash-lite

30 essays × 6 items = 180 calls. 0 failed after a retry. 0 reasons ran past one sentence.

## Against the human score, on the 1-10 scale

The human score is the mean of two raters, doubled.

| Item | Exact | Within 1 | Mean gap | Bias | r | QWK |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cohesion | 7% | 40% | 1.77 | -1.43 | 0.56 | 0.37 |
| syntax | 13% | 43% | 1.73 | -1.67 | 0.63 | 0.36 |
| vocabulary | 0% | 10% | 2.73 | -2.73 | 0.69 | 0.18 |
| phraseology | 3% | 13% | 2.70 | -2.70 | 0.61 | 0.16 |
| grammar | 7% | 40% | 1.93 | -1.87 | 0.69 | 0.41 |
| conventions | 7% | 30% | 1.93 | -1.80 | 0.72 | 0.43 |
| **all** | **6%** | **29%** | **2.13** | **-2.03** | **0.60** | **0.30** |

Bias above zero means the model scores higher than the people. r is the correlation, which ignores a constant offset; QWK is quadratic weighted kappa, which does not.

## How the scores spread

| Score | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Model | 4 | 14 | 64 | 42 | 22 | 29 | 4 | 1 | 0 | 0 |
| Human | 0 | 1 | 4 | 23 | 42 | 48 | 28 | 28 | 6 | 0 |

## With the average gap taken off

Adding 2 to every score (capped at 10): exact 32%, within 1 81%, mean gap 0.94, QWK 0.60. The shift is worked out on these same essays, so this is the best case, not a result.

## Reasons that mention spelling, capitals or punctuation

| Item | Reasons |
| --- | ---: |
| cohesion | 0 of 30 |
| syntax | 5 of 30 |
| vocabulary | 9 of 30 |
| phraseology | 0 of 30 |
| grammar | 10 of 30 |
| conventions | 29 of 30 |

## Against each rater, on the raters' own 1-5 scale

The model's 1-10 score is put back into its band (1-2 is level 1, 9-10 is level 5) and compared with each rater on their own. The last row is the two raters compared with each other, which is the bar to clear.

| Comparison | Exact | Within 1 | r | QWK |
| --- | ---: | ---: | ---: | ---: |
| Model vs rater 1 | 26% | 83% | 0.51 | 0.31 |
| Model vs rater 2 | 29% | 91% | 0.52 | 0.35 |
| Rater 1 vs rater 2 | 54% | 97% | 0.59 | 0.59 |

| Item | Model vs a rater, exact | Raters vs each other, exact | Model QWK | Raters QWK |
| --- | ---: | ---: | ---: | ---: |
| cohesion | 38% | 30% | 0.34 | 0.38 |
| syntax | 33% | 53% | 0.40 | 0.54 |
| vocabulary | 12% | 67% | 0.21 | 0.66 |
| phraseology | 15% | 53% | 0.19 | 0.58 |
| grammar | 35% | 57% | 0.40 | 0.70 |
| conventions | 33% | 67% | 0.49 | 0.68 |

## Where the model and the people disagree most

| Essay | Item | Model | Human | Gap | The model's reason |
| --- | --- | ---: | ---: | ---: | --- |
| 6CB7C1B3C548 | vocabulary | 3 | 8 | -5 | The vocabulary is very narrow and repetitive, relying on basic phrasing like 'get into trouble' and 'poor choice' alongside frequent spelling errors in word forms like 'introble' and 'design'. |
| 6CB7C1B3C548 | phraseology | 3 | 8 | -5 | The essay relies heavily on a narrow range of basic phrase patterns and frequent repetitions like 'get in trouble' and 'poor choice'. |
| 6CB7C1B3C548 | grammar | 3 | 8 | -5 | The essay contains many errors in grammar and usage, including persistent misspellings, missing apostrophes, run-on sentences, and incorrect verb forms like 'should of thought' and 'suppost'. |
| 1DE0E7C48DF0 | vocabulary | 3 | 7 | -4 | The vocabulary is very narrow and basic, featuring frequent spelling and word formation errors such as "hep", "wiling", and "Mangers" that limit the expression of ideas. |
| 1DE0E7C48DF0 | grammar | 3 | 7 | -4 | The essay contains many errors in grammar, usage, and mechanics, such as homophone confusion, missing capitalization, run-on sentences, and fragment structures. |
| 7A7D7BC53938 | grammar | 2 | 6 | -4 | The essay contains frequent and severe errors in grammar, word forms, and agreement throughout, such as 'Its a fact', 'possitive actitude', 'every single thinsg that you made', and 'fealing' instead of 'failing'. |
| 50611B996992 | syntax | 3 | 7 | -4 | The essay contains frequent sentence-structure problems, run-on constructions, and awkward clauses that hinder readability. |
| D84024335230 | phraseology | 3 | 7 | -4 | The essay relies on a very narrow range of basic phrases and formulaic chunks like 'at the same time' and 'one time', accompanied by frequent awkward phrasings and grammatical misuses. |
