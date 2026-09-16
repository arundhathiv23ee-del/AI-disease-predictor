# scripts/train_kidney.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("Downloading Chronic Kidney Disease dataset...")
url = "https://matthew-brett.github.io/dsfe2019/data/ckd_clean.csv"
df = pd.read_csv(url)

# Fill missing values instead of dropping rows to preserve samples
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].astype('category').cat.codes
    else:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)

# Save a local copy
df.to_csv('data/kidney.csv', index=False)

target_col = df.columns[-1] 
print(f"Target column identified as: {target_col}")
print(f"Total valid samples: {len(df)}")

X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("Training Kidney Disease ML Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

joblib.dump(model, 'models/model_kidney.pkl')
joblib.dump(scaler, 'models/scaler_kidney.pkl')

print("Success! Kidney model and scaler saved to /models directory.")