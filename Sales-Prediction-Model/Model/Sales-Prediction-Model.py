import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


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

#results
with open("model_evaluation.txt", "w") as f:
    f.write("Model Evaluation Results:\n")
    f.write(f"Model Coefficients: {model.coef_}\n")
    f.write(f"Intercept: {model.intercept_}\n")
    f.write(f"Mean Squared Error (MSE): {mse}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
    f.write(f"R-squared (R2): {r2}\n")

print("Model training and evaluation complete. Results saved to sales_model_prediction.txt.")
