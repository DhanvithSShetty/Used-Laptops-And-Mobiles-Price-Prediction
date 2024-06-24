import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(current_dir, "..", "datasets", "mobile_data.csv")

try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    print(f"Error: File not found at {dataset_path}. Please check the path.")
    exit(1)  # Exit the script if the file is not found

df.ffill(inplace=True)

X = df.drop("Original_Price", axis=1)
y = df["Original_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ["Age(Years)"]
categorical_features = ["Brand", "Model", "Condition", "Specifications"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Define the model pipeline with a random forest regressor
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# Define hyperparameters for tuning
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_features': ['auto', 'sqrt', 'log2'],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Train the model
grid_search.fit(X_train, y_train)

# Evaluate the Model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Perform cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mean = -cv_scores.mean()

# Convert negative MSE to positive for readability
cv_mean = abs(cv_mean)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Cross-validated Mean Squared Error: {cv_mean}")

# Save the Trained Model
model_path = os.path.join(current_dir, "mobile_price_model.pkl")
joblib.dump(best_model, model_path)
print(f"Model saved to: {model_path}")
