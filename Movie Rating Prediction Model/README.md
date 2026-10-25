# Movie Rating Prediction Model

Predicts the IMDb rating of Indian films from the IMDb Movies India dataset (`IMDb_Movies_India.csv`).

## How it works

[`Movie Rating Prediction Model.py`](Movie%20Rating%20Prediction%20Model.py) does the following:

- Cleans the raw columns: duration from text like "109 min" to a number, and votes and year to numbers. Films with no rating are dropped.
- Uses duration, votes, genre, director and the three lead actors as features. Missing numbers are filled with the median, and missing categories are marked as "missing" and one-hot encoded.
- Fits a random forest regressor on an 80/20 train-test split with a fixed random seed.
- Prints the mean absolute error and the root mean squared error on the test set, in rating points.

## Run it

From this folder:

```bash
python "Movie Rating Prediction Model.py"
```

## What I would do next

- Compare against a simple baseline (predicting the average rating) so the error has something to be judged against.
- Check which features carry the prediction. With one-hot encoded directors and actors, the model may be memorising names rather than learning anything general.
