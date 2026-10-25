# Statistical Modelling

Eight projects in Python with scikit-learn: five models I built while learning predictive modelling, two that test and forecast on synthetic South African service-desk data, and an inflation forecast on official Stats SA data. Each one sits in its own folder with its data, its script and a README that says what it does and what it found. Every model is compared with a simple baseline, so its score has something to be judged against.

| Project | Question | Method and result |
| --- | --- | --- |
| [Housing Prices Prediction Model](Housing%20Prices%20Prediction%20Model/) | What will a house in Ames, Iowa sell for? | Ridge regression on log price: RMSE 0.136 in cross-validation (a typical error of about 14%), against 0.407 for guessing the average |
| [Movie Rating Prediction Model](Movie%20Rating%20Prediction%20Model/) | Can genre, director, cast, length and votes predict an IMDb rating? | Random forest regression: off by 0.93 rating points on average, against 1.12 for guessing the average |
| [Sales Prediction Model](Sales-Prediction-Model/) | How much do TV, radio and newspaper advertising each add to sales? | Linear regression: cross-validated R² of 0.89 |
| [Titanic Survival Prediction Model](Titanic-Survival-Prediction-Model/) | Who survived the Titanic, from class, sex, age, family aboard, fare and port? | Random forest classifier: 82% accuracy in cross-validation, against 62% for predicting that nobody survived |
| [Iris Flower Classification Model](Iris-Flower-Classification-Model/) | Which species is a flower, from four measurements? | Decision tree: 95% accuracy in cross-validation |
| [Did the second sign-off slow refunds?](Refund-Sign-Off-Test/) | Did a new approval step make refunds slower, and is a bump in late complaints real? | Permutation tests and bootstrap intervals: refunds +37 points late and +2.0 working days; the complaints bump passes a significance test but is a fluke, and the next quarter shows it |
| [Request Volume Forecast](Request-Volume-Forecast/) | How many requests will a service desk get each day? | Poisson regression on weekday, month and SA public holidays, with a time-based split: 1.80 requests a day off, against 2.67 for "same as last week" |
| [CPI Inflation Forecast](CPI-Inflation-Forecast/) | Can a simple model forecast South African inflation better than "it stays where it is"? | Ridge regression on recent changes, backtested every month from 2010 on Stats SA's official rates: no significant gain over "no change" at 1, 3, 6 or 12 months |

The first five use public teaching datasets, the next two use invented data from [SA Turnaround](https://github.com/nurah-kamal/reporting-and-bi/tree/main/sa-turnaround), and the CPI forecast uses Statistics South Africa's CPI History, with the source and terms in its README. All the data is included in the folders.

## What these are, and what they are not

The first five are study projects on well-known teaching datasets, so they show that I can build, split, fit and evaluate a model properly. They are not original analysis. The two on synthetic data have an answer planted in advance, so they show whether a method finds what is really there and nothing else. The CPI forecast is on real data, and its honest answer is that a model did not beat the simplest guess. For the original analysis that data came from, see [SA Turnaround](https://github.com/nurah-kamal/reporting-and-bi/tree/main/sa-turnaround).

## Run one

Python 3.12 with the versions in [`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt
```

Then run the script from inside that project's folder, because each script reads its data with a path relative to the folder. The figures in each README come from a run with these versions.

## Licence

MIT.
