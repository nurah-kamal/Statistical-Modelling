# rubric-eval

[![rubric-eval](https://github.com/nurah-kamal/statistical-modelling/actions/workflows/rubric-eval.yml/badge.svg?branch=main&event=push)](https://github.com/nurah-kamal/statistical-modelling/actions/workflows/rubric-eval.yml) 36 tests

**Scores writing against a rubric with Gemini, one call per rubric item, then measures how far those scores agree with trained human raters. On 30 learner essays, the model ranked the essays almost as consistently as two raters ranked them against each other, but it scored nearly every essay a full level too low.**

```
$ npm run eval -- --client recorded --model gemini-3.5-flash-lite
...
180/180  D86C20117DC9 conventions: 4
gemini-3.5-flash-lite: exact 6%, within 1 29%, mean gap 2.13, QWK 0.30
Written: results/gemini-3.5-flash-lite.md and results/gemini-3.5-flash-lite.json
```

That replays the recorded run, so it needs no key. From the report it writes, each rater's scores compared on their own 1–5 scale:

| Comparison | Exact | Within 1 | r | QWK |
| --- | ---: | ---: | ---: | ---: |
| Model vs rater 1 | 26% | 83% | 0.51 | 0.31 |
| Model vs rater 2 | 29% | 91% | 0.52 | 0.35 |
| Rater 1 vs rater 2 | 54% | 97% | 0.59 | 0.59 |

The full report is [results/gemini-3.5-flash-lite.md](results/gemini-3.5-flash-lite.md).

## What it shows

| Question | Answer on this sample |
| --- | --- |
| **Does the model put the essays in the right order?** | Mostly. Its scores correlate 0.60 with the human score, and 0.51–0.52 with each single rater. The two raters correlate 0.59 with each other. |
| **Does it give the same scores?** | No. It sits 2.03 points below the human score on the 1–10 scale, so it matches exactly only 6% of the time and is within one point 29% of the time. |
| **Where is it furthest out?** | Vocabulary and phraseology, both 2.7 points low. Its ranking is best on conventions (correlation 0.72) and weakest on cohesion (0.56). |
| **How does it use the scale?** | Narrowly. 106 of its 180 scores are a 3 or a 4, and it gave a 7 or higher only 5 times. The people gave a 7 or higher 62 times. |
| **Does it keep the items apart?** | Not entirely. On grammar, 10 of its 30 reasons cite spelling, capitals or punctuation, and on vocabulary 9 of 30 do. Those belong to conventions. |
| **Is it better than counting?** | On ranking, clearly: a scorer that only counts words, sentence lengths and slips correlates 0.19, against the model's 0.60. On exact matches, no: the counter gets 17% because its scores sit in the middle of the scale, and the model gets 6%. |

## Why it disagrees

The model's reasons are specific and mostly accurate: it quotes real misspellings ("introble", "possitive actitude") and real repeated phrases. The disagreement is about where those faults put an essay, not whether they are there. Three things in the run point to why.

1. **It seems to read the bands as an absolute standard, where the raters read them against the learners in front of them.** The top band of the rubric describes near-native writing. The raters scored learners in grades 8 to 12 and used the upper half of the scale for that group: 62 of their scores are 7 or higher. The model's scores look like every essay was held against the written description alone, so a good learner essay with slips landed at 4 to 6.
2. **Surface errors leak into other items.** Each call asks about one item and says to ignore the rest, yet spelling turns up in a third of the grammar and vocabulary reasons. The biggest single gap is an essay the raters gave 8 for vocabulary, phraseology and grammar. The model gave it 3 on all three, and two of those three reasons cite its misspellings.
3. **The offset is steady, not random.** Adding 2 to every score lifts within-one agreement from 29% to 81% and kappa from 0.30 to 0.60, level with the 0.59 the two raters reach with each other. That 2 was worked out on these same 30 essays, so it is the best case rather than a result. But it shows that most of the gap is calibration, which can be fixed, rather than misreading, which can't easily be.

**What I'd do next, in order:** add one scored example essay per band to each prompt, so the model sees what a 7 looks like in this population; then fit the offset on half the essays and test it on the other half; then run the same comparison on the larger Gemini model, which the free tier caps at 20 requests a day (see below).

## The part worth copying

- **Compare the model with each rater, not just with the average.** Two trained people agree exactly only 54% of the time on this sample, and 52% across the 3,897 corpus essays that have both ratings. A model that matched a rater 54% of the time would be doing as well as a second rater, not scoring "54% accurate". Without that bar, every number here is uninterpretable.
- **Report ranking and calibration separately.** Correlation ignores a constant offset; kappa and exact match punish it. Here they tell different stories (0.60 against 0.30), and the difference is the finding.
- **One call per rubric item, with one sentence of evidence.** The reasons are what showed the spelling leak. A single call returning six numbers would have hidden it.
- **Every run is replayable.** Each reply is saved under a SHA-256 hash of its prompt, without the prompt itself. `--client recorded` rebuilds every figure in this README with no key and no network, and CI checks that the committed results still come out of the committed recording. Change the rubric or the prompt wording and the replay refuses, rather than quietly scoring something else.
- **A crude baseline is in the table.** The word-counting scorer beats the model on exact match. Showing that keeps the headline honest.

## How it is built

| | |
| --- | --- |
| **Rubric** | [rubrics/ellipse.yaml](rubrics/ellipse.yaml): six items, a 1–10 scale, and written guidance for each band of two points. It is adapted from the ELLIPSE rubric's five levels. `src/rubric.js` refuses a rubric with gaps, overlaps or bands off the scale. |
| **Prompt** | `src/prompt.js` builds one prompt per item: the item, its bands, the task the essay answered, and the essay inside a fence marked as text to be scored, not instructions. Gemini is asked for JSON with `score` and `reason`, at temperature 0. |
| **Calls** | `src/gemini.js` uses the REST API directly, with the key in a header. Calls are spaced out, a 429 or 503 is retried after the delay the API asks for, and a hung call times out after 90 seconds. A used-up daily quota stops the run with a message, and `--resume` carries on later from the saved replies. Error messages name the status only, never the API's text, which can echo the request. |
| **Checking replies** | A score outside 1–10, a non-integer or a missing reason is asked for once more, then recorded as a failure. Nothing is guessed. All 180 calls in the published run returned a valid score, and every reason was one sentence. |
| **Agreement** | `src/agreement.js`: exact match, within one point, mean gap, bias, Pearson correlation and quadratic weighted kappa. `src/report.js` adds the per-rater comparison, the score spread, the shifted view and the spelling count. |
| **Data** | 30 essays from the ELLIPSE corpus, six from each of five score bands, drawn with a fixed seed by `scripts/make-sample.js`. [data/README.md](data/README.md) covers the source, the licence and what was removed. |
| **Tests** | 36 tests with Node's built-in runner: rubric validation, prompt building, reply checking, the agreement maths against hand-worked cases, the Gemini client against a fake server (retries, timeouts, quota, the key never in the URL or an error), recording and replay, resuming, and the sample itself. |
| **CI** | `.github/workflows/rubric-eval.yml` in the repository root runs the tests, a dependency audit, and both replays with a check that the committed results haven't drifted. `secrets.yml` scans the whole repository's history on every push. Actions are pinned to exact commits. |

## Try it on your own machine

Node 22.13 or later.

```bash
cd rubric-eval
npm ci
npm test
npm run eval -- --client recorded --model gemini-3.5-flash-lite   # the published run, no key
npm run eval -- --client stub                                     # the word-counting baseline
```

To score with Gemini yourself, get a free key from Google AI Studio and put it in `.env`. Copy [.env.example](.env.example); `.env` is ignored by git.

```bash
npm run eval -- --client gemini --model gemini-3.5-flash-lite --record --label mine
```

Use `--limit 2` to try two essays first. To score something else, write a rubric in the same shape and point `--rubric` and `--data` at your own files. [rubric-builder](../rubric-builder/) ([live](https://nurah-kamal.github.io/statistical-modelling/rubric-builder/)) writes one on a single screen, checks it, and exports a JSON file `--rubric` reads as it is.

## Things worth knowing

**Which model, and why that one.** The published run is `gemini-3.5-flash-lite`, recorded on 25 October 2026. The plan was a larger model. `gemini-2.5-flash` is no longer offered to new keys, `gemini-3.8-flash` was refusing requests for high demand that evening, and `gemini-3.6-flash` allows 20 requests a day on the free tier, so a 180-call run would take nine days. The lite model is the one that could do the whole run on a free key. A larger model may well be better calibrated, and that is the next run worth doing.

**30 essays is a small sample.** It is enough to show a two-point offset that holds on every item, but not to rank the items against each other with any confidence. One essay more or less moves an item's exact-match rate by about three points.

**The human score is not always the raters' mean.** In 4 of the 180 item scores the raters were two points apart and the corpus published something else. [data/README.md](data/README.md) has the detail.

**The essays were written by real students.** They are published in the corpus without names or other identifying details. The sample also leaves out every essay where a name had to be removed, and every demographic column. Essay IDs are the corpus's own.

**The idea comes from work I do professionally.** It was rebuilt from scratch on an open corpus, and no work code, rubric or data is in this folder.

## Folders

```
bin/eval.js              the command line: stub, recorded or gemini
src/rubric.js            loads and checks a rubric
src/prompt.js            one prompt per item, and checking the reply
src/gemini.js            the API client: spacing, retries, timeouts, quota
src/clients.js           recording, replay, resume, and the word-counting baseline
src/score.js             every essay against every item
src/agreement.js         exact, within one, gap, bias, correlation, kappa
src/report.js            the comparison and the Markdown report
rubrics/ellipse.yaml     the rubric used here
data/                    the 30 essays with both raters' scores, and their source
recordings/              every reply from the published run, keyed by prompt hash
results/                 the reports and raw scores for the model and the baseline
scripts/make-sample.js   draws the sample from the corpus files
tests/                   36 tests
```

## Licence

The code is MIT, in the [repository root](../LICENSE). The essays, their scores and the adapted rubric are from the ELLIPSE corpus and stay under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); see [data/README.md](data/README.md).
