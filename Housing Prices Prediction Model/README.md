# Housing Prices Prediction Model

Predicts the sale price of houses in Ames, Iowa from the Ames Housing dataset (`DataSet/AmesHousing.csv`, 2,930 sales with about 80 features each).

## How it works

[`Model/HousingPredictionModel.py`](Model/HousingPredictionModel.py) does the following:

- Log-transforms the sale price, because prices are right-skewed and a log scale makes errors proportional rather than dominated by expensive houses.
- Splits the features into numbers and categories. Missing numbers are filled with the mean and then standardised. Missing categories are filled with the most common value and then one-hot encoded.
- Fits a Ridge regression (alpha 1.0) on an 80/20 train-test split with a fixed random seed.
- Prints the root mean squared error on the test set. Because the target is log price, the RMSE is in log units: an RMSE of 0.15 means predictions are typically about 15% off.
- Compares it with a baseline that predicts the average training price for every house, gives both errors in dollars, and repeats the fit with 5-fold cross-validation so the score does not rest on one split.

## Results

From a run with the versions in [`requirements.txt`](../requirements.txt):

| | Ridge regression | Baseline (always the average) |
| --- | --- | --- |
| RMSE on the test set (log price, 586 houses) | 0.119 | 0.433 |
| Mean absolute error on the test set | $15,070 | $63,733 |
| RMSE, 5-fold cross-validation | 0.136 (folds 0.119 to 0.154) | 0.407 |

An RMSE of 0.12 to 0.14 in log price means a typical prediction is roughly 12–15% away from the sale price, against roughly 40–50% for a guess of the average. The test split happens to be the easiest of the five folds, so the cross-validated 0.136 is the fairer figure.

Cautions: Ames sales from 2006 to 2010, in dollars, say nothing about any other market. Every column goes into the model, including ones only known at the time of sale (such as the sale condition) and the parcel ID read as a number, so a real pricing tool would do worse than this.

## Charts

- [`Visualisations/SalesPrice Distribution`](Visualisations/SalesPrice%20Distribution/): the distribution of sale prices, which shows the skew that the log transform corrects.
- [`Visualisations/Heatmap`](Visualisations/Heatmap/): correlations between the numeric features and sale price.

## Run it

From this folder:

```bash
python Model/HousingPredictionModel.py
```

## What I would do next

- Tune alpha with cross-validation instead of fixing it at 1.0, and compare against a plain linear regression.
- Drop the columns only known at the time of sale, and the IDs, and see how much the error grows.
