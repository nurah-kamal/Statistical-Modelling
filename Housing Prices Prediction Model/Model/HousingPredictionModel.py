# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
data = pd.read_csv('DataSet/AmesHousing.csv')
X = data.drop('SalePrice', axis=1)
y = np.log(data['SalePrice'])  # Log-transform the target to handle skewness

# Identify numerical and categorical columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'str']).columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean
    ('scaler', StandardScaler())  # Standardize numerical features
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with mode
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical features
])

# Combine preprocessors in a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create a pipeline with Ridge regression
ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))  # Regularized regression
])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline
ridge_pipeline.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = ridge_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Root Mean Squared Error (RMSE): {rmse}")

# Baseline to beat: predict the average log price of the training houses for every house
baseline = DummyRegressor(strategy='mean').fit(X_train, y_train)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline.predict(X_test)))
print(f"Baseline RMSE (always predict the mean): {baseline_rmse:.4f}")

# The same errors in dollars
mae_dollars = mean_absolute_error(np.exp(y_test), np.exp(y_pred))
baseline_mae_dollars = mean_absolute_error(np.exp(y_test), np.exp(baseline.predict(X_test)))
print(f"Mean absolute error in dollars: model {mae_dollars:,.0f}, baseline {baseline_mae_dollars:,.0f}")

# 5-fold cross-validation on all the data, so the score does not rest on one split
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse = -cross_val_score(ridge_pipeline, X, y, cv=kf, scoring='neg_root_mean_squared_error')
cv_baseline = -cross_val_score(DummyRegressor(strategy='mean'), X, y, cv=kf, scoring='neg_root_mean_squared_error')
print(f"Cross-validated RMSE: {cv_rmse.mean():.4f} (folds {np.round(cv_rmse, 4)})")
print(f"Cross-validated baseline RMSE: {cv_baseline.mean():.4f}")
