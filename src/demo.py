# demo.py — Interactive demo for diabetes prediction

import numpy as np
import xgboost as xgb
import joblib

# Load the trained model (assuming you saved it in main.py as model.joblib)
model = joblib.load("results/model_xgb.joblib")

print("=== Diabetes Risk Prediction Demo ===")

# Ask the user for input values
age = float(input("Enter age: "))
bmi = float(input("Enter BMI: "))
glucose = float(input("Enter blood glucose level: "))
hba1c = float(input("Enter HbA1c level: "))
gender = int(input("Gender (0 = Female, 1 = Male): "))
hypertension = int(input("Hypertension (0 = No, 1 = Yes): "))
heart_disease = int(input("Heart disease (0 = No, 1 = Yes): "))
smoking = int(input("Smoking history (0 = Never, 1 = Former, 2 = Current): "))

# Create a feature vector
features = np.array([[gender, age, hypertension, heart_disease,
                      smoking, bmi, hba1c, glucose]])

# Get probability and prediction
prob = model.predict_proba(features)[0][1]
pred = int(prob >= 0.21)  # use your tuned threshold

print(f"\nPredicted probability of diabetes: {prob:.2f}")
if pred == 1:
    print("⚠️ High risk of diabetes — recommend follow-up testing.")
else:
    print("✅ Low risk of diabetes.")

