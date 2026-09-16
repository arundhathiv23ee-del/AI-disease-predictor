# scripts/train_liver.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("Downloading Liver Disease dataset...")
# Official UCI Repository URL for the Indian Liver Patient Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"
columns = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 
           'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 
           'Albumin', 'Albumin_and_Globulin_Ratio', 'Target']

df = pd.read_csv(url, names=columns)

# Drop missing values (ILPD has a few missing in the A/G ratio column)
df = df.dropna()

# Convert categorical Gender to binary (Male: 1, Female: 0)
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

# The dataset target is 1 (Liver disease) and 2 (No liver disease). 
# We convert this to standard binary (1 = Risk, 0 = Safe).
df['Target'] = df['Target'].apply(lambda x: 1 if x == 1 else 0)

# Save a local copy
df.to_csv('data/liver.csv', index=False)

# Prepare Data
X = df.drop('Target', axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("Training Liver ML Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Serialize and save
joblib.dump(model, 'models/model_liver.pkl')
joblib.dump(scaler, 'models/scaler_liver.pkl')

print("Success! Liver model and scaler saved to /models directory.")