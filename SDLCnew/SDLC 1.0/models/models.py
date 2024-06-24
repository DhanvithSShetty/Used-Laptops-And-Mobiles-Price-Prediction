import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

# Get the absolute path to the directory where this script resides
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the datasets directory
dataset_path = os.path.join(current_dir, "..", "datasets", "laptop_data.csv")

# Load the dataset, handling potential FileNotFoundError
try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    print(f"Error: File not found at {dataset_path}. Please check the path.")
    exit(1)  # Exit the script if the file is not found

# Handle missing values (if any)
df.ffill(inplace=True)

# Define features and target
X = df.drop("Original_Price", axis=1)
y = df["Original_Price"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing pipeline
numeric_features = ["Age(Years)", "Battery Life (hrs)", "Screen Size (inches)"]
categorical_features = ["Brand", "Model", "Condition", "Specifications", "Storage Type"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Choose and Train a Model
model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))])

# Train the model
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = np.mean(np.abs(y_pred - y_test))
accuracy = 1 - mse / np.var(y_test)

# Print evaluation metrics
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Model Accuracy: {accuracy:.4f}")

# Print feature importances
regressor = model.named_steps['regressor']
if hasattr(regressor, 'feature_importances_'):
    importances = regressor.feature_importances_
    feature_names = numeric_features + list(preprocessor.transformers_[1][1].get_feature_names_out(categorical_features))
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    print(importance_df.sort_values(by='Importance', ascending=False))

# Save the Trained Model
model_path = os.path.join(current_dir, "laptop_price_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")
