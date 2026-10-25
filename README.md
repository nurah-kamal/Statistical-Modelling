# Statistical Modelling

Five models I built while learning predictive modelling in Python with scikit-learn. Each one sits in its own folder with its data, its script and a README that says what it does and what it found.

| Project | Question | Method |
| --- | --- | --- |
| [Housing Prices Prediction Model](Housing%20Prices%20Prediction%20Model/) | What will a house in Ames, Iowa sell for? | Ridge regression on every feature, with the price log-transformed |
| [Movie Rating Prediction Model](Movie%20Rating%20Prediction%20Model/) | Can genre, director, cast, length and votes predict an IMDb rating? | Random forest regression |
| [Sales Prediction Model](Sales-Prediction-Model/) | How much do TV, radio and newspaper advertising each add to sales? | Linear regression |
| [Titanic Survival Prediction Model](Titanic-Survival-Prediction-Model/) | Who survived the Titanic, from class, sex, age, family aboard, fare and port? | Random forest classifier, 80% accuracy on the test set |
| [Iris Flower Classification Model](Iris-Flower-Classification-Model/) | Which species is a flower, from four measurements? | Decision tree |

All five use public datasets that are included in the folders.

## What these are, and what they are not

These are study projects on well-known teaching datasets, so they show that I can build, split, fit and evaluate a model properly. They are not original analysis. For original analysis, see [SA Turnaround](https://github.com/nurah-kamal/sa-turnaround).

## Run one

Python 3 with pandas, numpy, scikit-learn and matplotlib:

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

Then run the script from inside that project's folder, because each script reads its data with a path relative to the folder.

## Licence

MIT.
