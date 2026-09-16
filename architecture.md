# architecture.md

## System Architecture: AI-Powered Multi-Disease Prediction System

**Architecture Overview**
The system employs a streamlined, Python-centric architecture utilizing Streamlit to merge frontend UI and backend execution into a single, cohesive application. This structure allows for rapid development within the 24-hour deadline while fulfilling all core machine learning inference and reporting requirements.

### 1. Presentation Layer (Frontend - Streamlit)
*   **User Input Interface:** The primary web application where users enter symptoms, age, lifestyle habits, and basic vitals[cite: 1].
*   **Results Dashboard:** An interactive layout that provides predictions for multiple diseases like diabetes, heart disease, liver disease, and kidney failure simultaneously[cite: 1].
*   **Appointment Module:** A simplified form component functioning as the doctor appointment booking interface[cite: 1] to capture patient scheduling requests.
*   **Download Center:** A Streamlit native download button linked to the dynamically generated PDF reports.

### 2. Application Logic Layer (Backend - Python)
*   **State Management:** Utilizes Streamlit's `st.session_state` to temporarily cache user inputs (vitals, symptoms) so data persists while the user navigates between the prediction dashboard and the appointment booking page.
*   **PDF Generation Engine:** A dedicated module using the `fpdf` library to run the report generation feature that creates a downloadable PDF health summary with risk scores and lifestyle recommendations[cite: 1].

### 3. Machine Learning Pipeline (Inference & Explainability)
*   **Model Registry:** A local directory containing serialized (`.pkl` or `.joblib`) versions of the separate ML models for each disease[cite: 1] previously trained on datasets from the UCI Machine Learning Repository and Kaggle[cite: 1].
*   **Data Preprocessor:** A scaling and encoding pipeline that transforms the raw Streamlit form inputs into the exact dimensional arrays required by the Scikit-Learn/XGBoost models.
*   **Explainability Engine:** A module powered by the `shap` library that generates visual plots on the fly to explain why it made each prediction using SHAP values so users trust the output[cite: 1].

### 4. Deployment Infrastructure
*   **Version Control:** Git repository hosted on GitHub containing the code, serialized models, and `requirements.txt`.
*   **Cloud Hosting:** Streamlit Community Cloud or Render connected directly to the GitHub repository to instantly deploy on a public URL so anyone can access it[cite: 1].
*   **Scalability:** Configured as a lightweight stateless application designed to serve the target users: health-conscious individuals, small clinics, and corporate wellness programs[cite: 1].