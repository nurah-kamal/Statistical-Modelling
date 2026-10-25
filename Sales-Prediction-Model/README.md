# Sales Prediction Model

How much do TV, radio and newspaper advertising each add to sales? A linear regression on the advertising dataset (`Dataset/advertising (1).csv`: 200 markets, with budgets in thousands and sales in thousands of units).

## How it works

[`Model/Sales-Prediction-Model.py`](Model/Sales-Prediction-Model.py) splits the data 80/20 with a fixed random seed, fits a linear regression on the three budgets, compares it with a baseline that predicts the average sales for every market, repeats the fit with 5-fold cross-validation, and writes the results to `Model/model_evaluation.txt`.

## Results

From [`Model/model_evaluation.txt`](Model/model_evaluation.txt):

| | Value |
| --- | --- |
| R² on the test set | 0.91 |
| RMSE on the test set | 1.71 (thousand units) |
| Baseline RMSE (always the average) | 5.65 |
| R², 5-fold cross-validation | 0.89 (folds 0.88 to 0.91) |
| RMSE, 5-fold cross-validation | 1.68, against 5.31 for the baseline |
| TV coefficient | 0.055 |
| Radio coefficient | 0.101 |
| Newspaper coefficient | 0.004 |

Each extra thousand spent on radio goes with about 101 more units sold, TV with about 55, and newspaper with almost none. The model's typical error is about a third of the baseline's, and the cross-validated R² of 0.89 shows the 0.91 on the test set was not a lucky split.

A caution: these are associations in one dataset, not proof that the spend caused the sales. Radio budgets may simply be higher in markets that were going to sell more anyway.

## Run it

From this folder:

```bash
python Model/Sales-Prediction-Model.py
```
