
# Titanic Survival Prediction Model

## Project Overview

This project aims to build a machine learning model to predict whether a passenger on the Titanic survived or not using the Titanic dataset. This is a classic beginner project in data science and machine learning, often used to demonstrate various techniques in data preprocessing, feature engineering, model training, and evaluation.

## Dataset

The dataset used for this project is the Titanic dataset, which contains information about individual passengers, including:

- **PassengerId**: Unique ID for each passenger
- **Pclass**: Ticket class (1st, 2nd, 3rd)
- **Name**: Name of the passenger (not used in the model)
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number (not used in the model)
- **Fare**: Fare paid by the passenger
- **Cabin**: Cabin number (not used in the model)
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- **Survived**: Survival status (0 = No, 1 = Yes)

### Data Source
[Link to the Titanic Dataset](https://www.kaggle.com/c/titanic/data)

## Libraries Used

The following libraries are utilized in this project:

- `pandas`: For data manipulation and analysis
- `numpy`: For numerical computing
- `scikit-learn`: For machine learning algorithms and metrics
- `seaborn`: For data visualization
- `matplotlib`: For creating static, animated, and interactive visualizations

## Features

- Data cleaning and preprocessing, including handling missing values and encoding categorical variables.
- Splitting the dataset into training and testing sets for model evaluation.
- Implementation of a Random Forest Classifier for predicting survival.
- Model evaluation using accuracy score and classification report.

## Model Performance

After training and evaluating the model, the following metrics were obtained:

- **Accuracy:** 80%
- **Precision:**
  - Class 0 (Did not survive): 0.82
  - Class 1 (Survived): 0.76
- **Recall:**
  - Class 0: 0.84
  - Class 1: 0.74
- **F1-Score:**
  - Class 0: 0.83
  - Class 1: 0.75
- **Support:**
  - Class 0: 105
  - Class 1: 74
- **Macro Average:**
  - Precision: 0.79
  - Recall: 0.79
  - F1-Score: 0.79
- **Weighted Average:**
  - Precision: 0.80
  - Recall: 0.80
  - F1-Score: 0.80

## Insights

- The model achieved an accuracy of 80%, indicating good predictive performance.
- Precision for non-survivors is higher than that for survivors, suggesting that the model is more reliable in predicting those who did not survive.
- The recall for survivors is lower than for non-survivors, indicating that some actual survivors were not correctly identified.




