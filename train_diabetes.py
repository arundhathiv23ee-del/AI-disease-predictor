# scripts/train_diabetes.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("Downloading PIMA Indians Diabetes dataset...")
# Using a raw GitHub URL to bypass manual Kaggle downloads
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# Save a local copy to our data folder
df.to_csv('data/diabetes.csv', index=False)

# Prepare Data
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features (Critical for medical vitals)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("Training Diabetes ML Model...")
# Using Random Forest for high baseline accuracy without hyperparameter tuning
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Serialize and save the model and scaler
joblib.dump(model, 'models/model_diabetes.pkl')
joblib.dump(scaler, 'models/scaler_diabetes.pkl')

print("Success! Model and Scaler saved to /models directory.")