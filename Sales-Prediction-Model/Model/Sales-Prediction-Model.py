import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os


#Loading the dataset
data = pd.read_csv('Dataset/advertising (1).csv')

#Seperating
x = data [['TV', 'Radio', 'Newspaper']]
y = data['Sales']

#Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Initializing the model
model = LinearRegression()

#Training the model
model.fit(x_train, y_train)

#Predicting
y_pred = model.predict(x_test)

#Evaluating the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

#Baseline to beat: predict the average training sales for every market
baseline_rmse = np.sqrt(mean_squared_error(y_test, DummyRegressor(strategy='mean').fit(x_train, y_train).predict(x_test)))

#5-fold cross-validation on all 200 markets
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(LinearRegression(), x, y, cv=kf, scoring='r2')
cv_rmse = -cross_val_score(LinearRegression(), x, y, cv=kf, scoring='neg_root_mean_squared_error')
cv_baseline_rmse = -cross_val_score(DummyRegressor(strategy='mean'), x, y, cv=kf, scoring='neg_root_mean_squared_error')

#results
#saved next to this script, where the README links to it
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_evaluation.txt"), "w") as f:
    f.write("Model Evaluation Results:\n")
    f.write(f"Model Coefficients: {model.coef_}\n")
    f.write(f"Intercept: {model.intercept_}\n")
    f.write(f"Mean Squared Error (MSE): {mse}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
    f.write(f"R-squared (R2): {r2}\n")
    f.write(f"Baseline RMSE (always predict the mean): {baseline_rmse}\n")
    f.write(f"Cross-validated R2 (5 folds): {cv_r2.mean()} (folds {np.round(cv_r2, 3)})\n")
    f.write(f"Cross-validated RMSE (5 folds): {cv_rmse.mean()}\n")
    f.write(f"Cross-validated baseline RMSE: {cv_baseline_rmse.mean()}\n")

print("Model training and evaluation complete. Results saved to Model/model_evaluation.txt.")
