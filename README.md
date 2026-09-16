# 🏥 AI-Powered Multi-Disease Prediction System

An enterprise-grade, unified clinical diagnostic platform that leverages Machine Learning to predict the risk of multiple major diseases simultaneously based on a patient's master health record.

This system moves beyond basic black-box predictions by integrating SHAP (SHapley Additive exPlanations) to provide transparent, feature-level rationale for every AI assessment, concluding with a downloadable clinical PDF report.

---

## ✨ Key Features

- **Simultaneous Multi-Model Inference:** Evaluates patient vitals against four distinct, finely tuned Random Forest models concurrently:
  - 🩸 Diabetes (Trained on PIMA Indians dataset)
  - 🫀 Heart Disease (Trained on Cleveland dataset)
  - ⚗️ Liver Disease (Trained on ILPD dataset)
  - 🫘 Chronic Kidney Disease (Trained on CKD dataset)

- **Unified Master Pipeline:** A clean, accessible UI that accepts a single comprehensive patient profile and dynamically routes specific features to their respective models in the background.

- **Explainable AI (XAI):** Automatically generates SHAP feature-impact charts for any flagged high-risk diseases, explaining exactly why the model made its prediction (e.g., highlighting that elevated glucose and BMI were the primary risk drivers).

- **Clinical PDF Generation:** Utilizes fpdf2 to dynamically compile the patient's vitals, multi-disease risk status, and the SHAP explainability charts into a professional, downloadable medical summary.

---

## 🛠️ Technology Stack

- **Frontend & Routing:** Streamlit
- **Machine Learning:** Scikit-Learn (Random Forest Classifiers), Pandas, NumPy
- **Explainability & Visualization:** SHAP, Matplotlib
- **Document Generation:** FPDF2
- **Serialization:** Joblib

---

## 🚀 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/arundhathiv23ee-del/AI-disease-predictor.git
cd AI-disease-predictor
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
AI-Disease-Predictor/
│
├── app.py                  # Main Streamlit application and UI routing
├── requirements.txt        # Project dependencies
├── README.md                # Project documentation
│
├── models/                 # Serialized ML models and scalers (.pkl)
│   ├── model_diabetes.pkl
│   ├── scaler_diabetes.pkl
│   └── ...                 # (Heart, Liver, Kidney models)
│
├── scripts/                 # Training scripts for dataset ingestion and model generation
│   ├── train_diabetes.py
│   ├── train_heart.py
│   ├── train_liver.py
│   └── train_kidney.py
│
└── utils/                    # Helper utilities
    └── pdf_generator.py      # FPDF2 clinical report builder
```
