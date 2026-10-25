# Movie Rating Prediction Model

Predicts the IMDb rating of Indian films from the IMDb Movies India dataset (`IMDb_Movies_India.csv`).

## How it works

[`Movie Rating Prediction Model.py`](Movie%20Rating%20Prediction%20Model.py) does the following:

- Cleans the raw columns: duration from text like "109 min" to a number, and votes and year to numbers. Films with no rating are dropped.
- Uses duration, votes, genre, director and the three lead actors as features. Missing numbers are filled with the median, and missing categories are marked as "missing" and one-hot encoded.
- Fits a random forest regressor on an 80/20 train-test split with a fixed random seed.
- Prints the mean absolute error and the root mean squared error on the test set, in rating points.
- Compares them with a baseline that predicts the average training rating for every film, and repeats the fit with 5-fold cross-validation.

One bug fixed: vote counts of 1,000 or more are written with a comma ("1,086"), and the first version turned all 1,371 of them into missing values, which were then filled with the median. The script now strips the comma first.

## Results

7,919 of the 15,509 films have a rating. From a run with the versions in [`requirements.txt`](../requirements.txt):

| | Random forest | Baseline (always the average) |
| --- | --- | --- |
| MAE on the test set | 0.91 | 1.11 |
| RMSE on the test set | 1.20 | 1.36 |
| MAE, 5-fold cross-validation | 0.93 (folds 0.90 to 0.96) | 1.12 |

The model is off by about 0.9 of a rating point on a typical film, against 1.1 for always guessing the average of about 5.8. That is a real but small gain: it removes about a sixth of the error, and most of what makes a film rate well is not in these columns.

Cautions: the forest gets thousands of one-hot columns for directors and actors, so part of its gain may be memorising names that appear in both the training and test films. It would do worse on films by new directors. The run takes several minutes because of the cross-validation.

## Run it

From this folder:

```bash
python "Movie Rating Prediction Model.py"
```

## What I would do next

- Check which features carry the prediction, and test on films whose director never appears in the training set.
- Replace the one-hot names with something that generalises, such as a director's average rating on earlier films only.
