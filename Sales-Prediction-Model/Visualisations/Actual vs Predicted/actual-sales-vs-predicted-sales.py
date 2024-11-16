import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Loading the dataset
data = pd.read_csv('Dataset/advertising (1).csv')

X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Training 
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#plotting the graph
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='b')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.savefig("actual_vs_predicted_sales.png")
plt.close()

#printing the visualisation
print("Visualisation saved as 'actual_vs_predicted_sales.png'.")