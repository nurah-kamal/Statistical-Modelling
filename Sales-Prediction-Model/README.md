# Sales Prediction Model

How much do TV, radio and newspaper advertising each add to sales? A linear regression on the advertising dataset (`Dataset/advertising (1).csv`: 200 markets, with budgets in thousands and sales in thousands of units).

## How it works

[`Model/Sales-Prediction-Model.py`](Model/Sales-Prediction-Model.py) splits the data 80/20 with a fixed random seed, fits a linear regression on the three budgets, and writes the results to `model_evaluation.txt`.

## Results

From [`Model/model_evaluation.txt`](Model/model_evaluation.txt):

| | Value |
| --- | --- |
| R² on the test set | 0.91 |
| RMSE | 1.71 (thousand units) |
| TV coefficient | 0.055 |
| Radio coefficient | 0.101 |
| Newspaper coefficient | 0.004 |

Each extra thousand spent on radio goes with about 101 more units sold, TV with about 55, and newspaper with almost none. The model explains about 91% of the variation in sales on data it did not see.

A caution: these are associations in one dataset, not proof that the spend caused the sales. Radio budgets may simply be higher in markets that were going to sell more anyway.

## Run it

From this folder:

```bash
python Model/Sales-Prediction-Model.py
```
