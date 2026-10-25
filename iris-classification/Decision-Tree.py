import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import joblib

iris_data = pd.read_csv("IRIS.csv")
x = iris_data.drop(columns=['species'])
y = iris_data['species']

model = joblib.load('iris_flower_selection_model.pkl')
model.fit(x, y)

plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=x.columns, class_names=y.unique())
plt.title('Decision Tree Visualization for IRIS Flower DataSet')

plt.savefig('iris_decision_tree.png')
plt.close()
