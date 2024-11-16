# Iris Flower Classification Model

## Project Overview

This project aims to build a machine learning model to classify iris flowers into three species: Iris-setosa, Iris-versicolor, and Iris-virginica, using the Iris dataset. 

## Dataset

The dataset used for this project is the Iris dataset, which contains information about individual iris flowers, including:

- **Sepal Length**: Length of the sepal (in cm)
- **Sepal Width**: Width of the sepal (in cm)
- **Petal Length**: Length of the petal (in cm)
- **Petal Width**: Width of the petal (in cm)
- **Species**: Species of the iris flower (Iris-setosa, Iris-versicolor, Iris-virginica)

### Data Source

The Iris dataset can be found in the `IRIS.csv` file within this repository.

## Libraries Used

The following libraries are utilized in this project:

- **pandas**: For data manipulation and analysis
- **numpy**: For numerical computing
- **scikit-learn**: For machine learning algorithms and metrics
- **matplotlib**: For creating static, animated, and interactive visualizations
- **joblib**: For saving and loading the trained model

## Features

- Data cleaning and preprocessing, including handling missing values and encoding categorical variables.
- Splitting the dataset into training and testing sets for model evaluation.
- Implementation of a Decision Tree Classifier for predicting the species of iris flowers.
- Model evaluation using accuracy score, classification report, and confusion matrix.

## Model Performance

After training and evaluating the model, the following metrics were obtained:

- **Accuracy**: 1.00 (100%)

### Classification Report


                  precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        10
Iris-versicolor       1.00      1.00      1.00         9
 Iris-virginica       1.00      1.00      1.00        11

       accuracy                           1.00        30
      macro avg       1.00      1.00      1.00        30
   weighted avg       1.00      1.00      1.00        30

## Confusion Matrix

[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]

## Insights

- The model achieved an accuracy of 100%, indicating perfect predictive performance on the test dataset.
- The classification report shows that all precision, recall, and F1-scores for each species are 1.00, reflecting the model's ability to classify all samples correctly.
- The confusion matrix confirms that there were no misclassifications among the iris species.

## Conclusion

This project demonstrates the application of machine learning in classification tasks using the Iris dataset. 
The achieved accuracy of 100% indicates the model's effectiveness in classifying the iris flowers based on their measurements.

## Requirements

To run this project, you will need the following Python packages:

- Python 3.x
- scikit-learn
- pandas
- matplotlib
- joblib


