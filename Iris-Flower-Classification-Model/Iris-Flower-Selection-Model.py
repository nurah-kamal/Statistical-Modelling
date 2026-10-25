from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Load the dataset
iris_data = pd.read_csv("IRIS.csv")

# Splitting data set into x and y variables 
x = iris_data.drop(columns=['species'])  
y = iris_data['species'] 

# Splitting data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Decision tree classifier
model = DecisionTreeClassifier(random_state=42)

#Validating 
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, x, y, cv=kf, scoring='accuracy')

# Training 
model.fit(x_train, y_train)

# Predicting 
y_pred = model.predict(x_test)

# Evaluating 
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)
confusion_matrix_result = confusion_matrix(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report_result)
print("Confusion Matrix:\n", confusion_matrix_result)

#saving model
import joblib
joblib.dump(model, 'iris_flower_selection_model.pkl')
