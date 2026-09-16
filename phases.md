# phases.md

## AI-Powered Multi-Disease Prediction System - 24-Hour MVP Plan

**Objective:** Rapidly build and deploy a web application where users enter symptoms, age, lifestyle habits, and basic vitals to get predictions for multiple diseases like diabetes, heart disease, liver disease, and kidney failure simultaneously[cite: 1].

### Phase 1: Environment & Data Acquisition (Hours 1-3)
*   Initialize the Git repository and Python virtual environment (`requirements.txt`).
*   Procure pre-cleaned datasets for target diseases from the UCI Machine Learning Repository and Kaggle[cite: 1].
*   Standardize input features (age, vitals, lifestyle habits) across datasets to ensure a unified user input pipeline[cite: 1].

### Phase 2: Model Training & Serialization (Hours 4-8)
*   Train separate ML models for each disease[cite: 1].
*   Utilize rapid-training algorithms like Scikit-Learn's Random Forest or XGBoost to establish high-accuracy baselines without time-consuming hyperparameter tuning.
*   Serialize and export the trained models using `pickle` or `joblib` for instant frontend inference.

### Phase 3: Explainability Engine (Hours 9-11)
*   Integrate the `shap` library into the prediction pipeline.
*   Configure the system to explain why it made each prediction using SHAP values so users trust the output[cite: 1].
*   Format the SHAP outputs as static plots to be passed directly to the UI.

### Phase 4: Streamlit Web Application (Hours 12-17)
*   Construct the core frontend using Streamlit to rapidly capture user inputs.
*   Integrate the pickled models to display the simultaneous risk predictions.
*   Implement a simplified doctor appointment booking interface[cite: 1] using Streamlit form components to collect name, preferred date, and specialist type.

### Phase 5: PDF Report Generation (Hours 18-20)
*   Integrate the `fpdf` Python library.
*   Build the report generation feature that creates a downloadable PDF health summary with risk scores and lifestyle recommendations[cite: 1].
*   Link the generated PDF to a Streamlit download button.

### Phase 6: Deployment & Final Polish (Hours 21-24)
*   Deploy on a public URL so anyone can access it[cite: 1] using Streamlit Community Cloud (fastest) or Render.
*   Finalize the `README.md` and documentation, explicitly noting that the target users are health-conscious individuals, small clinics, and corporate wellness programs[cite: 1].