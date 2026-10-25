# Housing Prices Prediction Model

Predicts the sale price of houses in Ames, Iowa from the Ames Housing dataset (`DataSet/AmesHousing.csv`, 2,930 sales with about 80 features each).

## How it works

[`Model/HousingPredictionModel.py`](Model/HousingPredictionModel.py) does the following:

- Log-transforms the sale price, because prices are right-skewed and a log scale makes errors proportional rather than dominated by expensive houses.
- Splits the features into numbers and categories. Missing numbers are filled with the mean and then standardised. Missing categories are filled with the most common value and then one-hot encoded.
- Fits a Ridge regression (alpha 1.0) on an 80/20 train-test split with a fixed random seed.
- Prints the root mean squared error on the test set. Because the target is log price, the RMSE is in log units: an RMSE of 0.15 means predictions are typically about 15% off.

## Charts

- [`Visualisations/SalesPrice Distribution`](Visualisations/SalesPrice%20Distribution/): the distribution of sale prices, which shows the skew that the log transform corrects.
- [`Visualisations/Heatmap`](Visualisations/Heatmap/): correlations between the numeric features and sale price.

## Run it

From this folder:

```bash
python Model/HousingPredictionModel.py
```

## What I would do next

- Compare against a plain linear regression and a tuned alpha with cross-validation.
- Report the error back in rands or dollars, not only in log units, so it reads for a non-technical audience.
