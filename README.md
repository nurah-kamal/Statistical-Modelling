# Statistical Modelling

Ten projects. Eight are in Python with scikit-learn: five models I built while learning predictive modelling, two that test and forecast on synthetic South African service-desk data, and an inflation forecast on official Stats SA data. The ninth, in Node, measures how well a language model's rubric scores agree with trained human raters, and the tenth, in React, is a page for writing the rubrics it scores with. Each one sits in its own folder with its data, its script and a README that says what it does and what it found. Every model is compared with a simple baseline, so its score has something to be judged against.

| Project | Question | Method and result |
| --- | --- | --- |
| [Housing Prices Prediction Model](housing-prices/) | What will a house in Ames, Iowa sell for? | Ridge regression on log price: RMSE 0.136 in cross-validation (a typical error of about 14%), against 0.407 for guessing the average |
| [Movie Rating Prediction Model](movie-ratings/) | Can genre, director, cast, length and votes predict an IMDb rating? | Random forest regression: off by 0.93 rating points on average, against 1.12 for guessing the average |
| [Sales Prediction Model](advertising-sales/) | How much do TV, radio and newspaper advertising each add to sales? | Linear regression: cross-validated R² of 0.89 |
| [Titanic Survival Prediction Model](titanic-survival/) | Who survived the Titanic, from class, sex, age, family aboard, fare and port? | Random forest classifier: 82% accuracy in cross-validation, against 62% for predicting that nobody survived |
| [Iris Flower Classification Model](iris-classification/) | Which species is a flower, from four measurements? | Decision tree: 95% accuracy in cross-validation |
| [Did the second sign-off slow refunds?](refund-sign-off-test/) | Did a new approval step make refunds slower, and is a bump in late complaints real? | Permutation tests and bootstrap intervals: refunds +37 points late and +2.0 working days; the complaints bump passes a significance test but is a fluke, and the next quarter shows it |
| [Request Volume Forecast](request-volume-forecast/) | How many requests will a service desk get each day? | Poisson regression on weekday, month and SA public holidays, with a time-based split: 1.80 requests a day off, against 2.67 for "same as last week" |
| [CPI Inflation Forecast](cpi-inflation-forecast/) | Can a simple model forecast South African inflation better than "it stays where it is"? | Ridge regression on recent changes, backtested every month from 2010 on Stats SA's official rates: no significant gain over "no change" at 1, 3, 6 or 12 months |
| [rubric-eval](rubric-eval/)<br>[![rubric-eval](https://github.com/nurah-kamal/statistical-modelling/actions/workflows/rubric-eval.yml/badge.svg?branch=main&event=push)](https://github.com/nurah-kamal/statistical-modelling/actions/workflows/rubric-eval.yml) 36 tests | Does Gemini score learner essays the way trained raters do? | One call per rubric item on 30 ELLIPSE essays: it ranks them almost as consistently as two raters agree with each other (correlation 0.51–0.52 with each rater, against 0.59 between them), but scores 2 points lower on a 10-point scale |
| [rubric-builder](rubric-builder/)<br>[![rubric-builder](https://github.com/nurah-kamal/statistical-modelling/actions/workflows/rubric-builder.yml/badge.svg?branch=main&event=push)](https://github.com/nurah-kamal/statistical-modelling/actions/workflows/rubric-builder.yml) 14 tests | Can a marker write a rubric that rubric-eval will read, without touching YAML? | A one-screen React tool that checks the rubric as it is written and exports rubric-eval's exact format; a test round-trips the real ELLIPSE rubric byte for byte. [Live](https://nurah-kamal.github.io/statistical-modelling/rubric-builder/) |

The first five use public teaching datasets, the next two use invented data from [SA Turnaround](https://github.com/nurah-kamal/reporting/tree/main/sa-turnaround), the CPI forecast uses Statistics South Africa's CPI History, with the source and terms in its README, and rubric-eval uses 30 essays from the ELLIPSE corpus (CC BY-NC-SA 4.0). All the data is included in the folders.

## What these are, and what they are not

The first five are study projects on well-known teaching datasets, so they show that I can build, split, fit and evaluate a model properly. They are not original analysis. The two on synthetic data have an answer planted in advance, so they show whether a method finds what is really there and nothing else. The CPI forecast is on real data, and its honest answer is that a model did not beat the simplest guess. For the original analysis that data came from, see [SA Turnaround](https://github.com/nurah-kamal/reporting/tree/main/sa-turnaround).

The calls behind how these are measured and laid out are in [DECISIONS.md](DECISIONS.md).

## Run one

Python 3.12 with the versions in [`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt
```

Then run the script from inside that project's folder, because each script reads its data with a path relative to the folder. The figures in each README come from a run with these versions.

rubric-eval is Node 22.13 or later instead: `cd rubric-eval`, `npm ci`, `npm test`. Its README says how to replay the published run without a key. rubric-builder is Node too, and its README says how to run it.

## Licence

MIT.
