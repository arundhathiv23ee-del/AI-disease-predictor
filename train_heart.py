# scripts/train_heart.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("Downloading Heart Disease dataset...")
# Using the official UCI Machine Learning Repository URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ['Age', 'Sex', 'ChestPain', 'RestingBP', 'Cholesterol', 'FastingBloodSugar', 
           'RestingECG', 'MaxHeartRate', 'ExerciseAngina', 'Oldpeak', 'Slope', 'MajorVessels', 'Thal', 'Target']

# The dataset uses '?' for missing values
df = pd.read_csv(url, names=columns, na_values='?')

# Drop missing values to ensure clean training data
df = df.dropna()

# Save a local copy
df.to_csv('data/heart.csv', index=False)

# Prepare Data 
# The Cleveland dataset target is 0 (no presence) to 4. We convert this to binary (0 = safe, 1 = risk).
X = df.drop('Target', axis=1)
y = (df['Target'] > 0).astype(int) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("Training Heart Disease ML Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Serialize and save
joblib.dump(model, 'models/model_heart.pkl')
joblib.dump(scaler, 'models/scaler_heart.pkl')

print("Success! Heart model and scaler saved to /models directory.")