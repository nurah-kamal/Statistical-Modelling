# Titanic Survival Prediction Model

Who survived the Titanic? A random forest classifier on the Titanic passenger list ([`Titanic-Dataset.csv`](Titanic-Dataset.csv), 891 passengers, the training file from [Kaggle's Titanic competition](https://www.kaggle.com/c/titanic/data)).

## How it works

[`Survival_Prediction_Model.py`](Survival_Prediction_Model.py) does the following:

- Drops the ID, name, ticket and cabin columns.
- Fills missing ages with the median age and the two missing ports with the most common port. Sex becomes 0 or 1, and the port is one-hot encoded.
- Uses class, sex, age, siblings or spouse aboard, parents or children aboard, fare and port as features.
- Fits a random forest (100 trees) on an 80/20 train-test split with a fixed random seed, and prints the accuracy and the classification report.
- Compares it with a baseline that predicts the most common outcome (did not survive) for everyone, and repeats the fit with stratified 5-fold cross-validation.

One fix: the missing values were filled with `fillna(..., inplace=True)` on a single column. Under pandas 3 that no longer changes the data frame, so the script now assigns the filled column back.

## Results

From a run with the versions in [`requirements.txt`](../requirements.txt):

| | Random forest | Baseline (nobody survived) |
| --- | --- | --- |
| Accuracy on the test set (179 passengers) | 80% | 59% |
| Accuracy, 5-fold cross-validation | 82.3% (folds 78.7% to 85.4%) | 61.6% |

On the test set:

```
              precision    recall  f1-score   support
           0       0.82      0.84      0.83       105
           1       0.76      0.74      0.75        74
```

The model gets about 4 in 5 passengers right, against 3 in 5 for assuming nobody survived. It is better at spotting those who died than those who survived: about a quarter of the survivors in the test set were predicted to have died.

Cautions: the median age is worked out from all 891 passengers before the split, so the test set leaks slightly into training. Most of the signal is sex and class, which a two-line rule ("women and first class survived") already captures, so the forest's gain over that simple rule would be worth measuring.

## Run it

From this folder:

```bash
python Survival_Prediction_Model.py
```
