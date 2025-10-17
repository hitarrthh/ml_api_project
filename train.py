import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import r2_score 
import joblib
import os

print("Script started: Training model...")
os.makedirs("models", exist_ok=True)
try:
    df = pd.read_csv("data/california_housing.csv")
except FileNotFoundError:
    print("Error: data/california_housing.csv not found.")
    print("Please run the data generation command.")
    exit()

print("Data loaded successfully.")
features = df.columns.drop('MedHouseVal')
X = df[features]
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data preprocessed (scaled).")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("Model trained.")
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
print(f"Model R-squared (R2): {r2:.4f}")

pipeline = {
    'scaler': scaler,
    'model': model,
    'features': list(features) 
}
joblib.dump(pipeline, "models/model_pipeline.joblib")

print(f"Model and scaler saved to models/model_pipeline.joblib")
print("Script finished.")