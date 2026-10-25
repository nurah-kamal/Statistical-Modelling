# Decisions

The calls behind how this repository is built. Each one says what was decided, why, and when. Where no reason was written down at the time, it says what the choice does instead of inventing one after the fact.

## Measuring

**Every model is compared with a simple baseline.** A score means little alone. 82% accuracy on the Titanic data reads differently next to the 62% you'd get by predicting that nobody survived. Each README gives both.

**A result that doesn't beat the baseline is reported as it is.** *25 October 2026.* The CPI forecast doesn't beat "inflation stays where it is", and its README says so rather than dropping the project.

**Projects on synthetic data have the answer planted in advance.** *25 October 2026.* The refund test and the volume forecast use data where the real effect is known. That checks whether a method finds what is there, and also that it doesn't find what isn't.

**A language model is measured against people, not against a single "right" answer.** *25 October 2026.* [rubric-eval](rubric-eval/) compares Gemini's scores with each of two trained raters and with how often those two raters agree with each other. Ranking and calibration are reported separately, because here they tell different stories.

**Published figures can be replayed.** *25 October 2026.* rubric-eval saves every model reply, and its workflow replays them and fails if the results in the folder stop matching. The README's numbers are checked on every push, without a key.

## How the repository is organised

**One repository per field, one folder per project.** *25 October 2026.* Modelling lives here. Reporting, BI and analysis have their own repositories: [reporting](https://github.com/nurah-kamal/reporting), [business-intelligence](https://github.com/nurah-kamal/business-intelligence) and [data-analysis](https://github.com/nurah-kamal/data-analysis).

**Folders have lowercase hyphenated names.** *25 October 2026.* For example, `housing-prices` rather than `Housing Prices Prediction Model`. Paths and links need no escaping.

**A project in another language keeps its own checks.** *25 October 2026.* rubric-eval is Node, not Python. It has its own workflow, filtered to its folder, so it runs only when that folder changes.

## Data and writing

**Public, synthetic or openly licensed data only, each with its source and terms in the project's README.** *25 October 2026.* The ELLIPSE essays are CC BY-NC-SA 4.0 and keep that licence inside this MIT repository; the CPI data's terms are in its folder.

**No people in any data that I made.** *25 October 2026.* The synthetic sets have no names, real or invented. The essays in rubric-eval were written by students and published anonymised by the corpus; the sample also leaves out any essay that needed a name removed.

**READMEs describe only what is in the repository.** *25 October 2026.* Older READMEs here claimed a model that didn't exist and figures that didn't match the code. Now every figure comes from a run, and a project with no run yet states no figure.

**Commits are one short line.** *Since 30 September 2026.* Explanations go in the READMEs, not in commit messages.
