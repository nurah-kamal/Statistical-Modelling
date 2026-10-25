#Imported the necessary libraries 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#accessing and reading the data from the given dataset
data = pd.read_csv('Titanic-Dataset.csv')
data.head()

#checking for null values
data.isnull().sum()

#cleaning data and removing unnecessary columns
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

#Filling in any missing values
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0]) 

#Converting categorical variables to numerical
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

#Splitting the data into features and target
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#evalutaion 
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

#baseline to beat: predict that nobody survived (the majority class)
baseline = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
print(f"Baseline accuracy (predict the majority class): {accuracy_score(y_test, baseline.predict(X_test)):.2f}")

#5-fold cross-validation on all 891 passengers
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X, y, cv=skf, scoring='accuracy')
cv_baseline = cross_val_score(DummyClassifier(strategy='most_frequent'), X, y, cv=skf, scoring='accuracy')
print(f"Cross-validated accuracy: {cv_scores.mean():.3f} (folds {np.round(cv_scores, 3)})")
print(f"Cross-validated baseline accuracy: {cv_baseline.mean():.3f}")

