# app.py
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
from utils.pdf_generator import generate_report

st.set_page_config(page_title="Multi-Disease Diagnostic Platform", layout="wide")

# --- Load All Models & Scalers ---
@st.cache_resource
def load_all_assets():
    return {
        'diabetes': (joblib.load('models/model_diabetes.pkl'), joblib.load('models/scaler_diabetes.pkl')),
        'heart': (joblib.load('models/model_heart.pkl'), joblib.load('models/scaler_heart.pkl')),
        'liver': (joblib.load('models/model_liver.pkl'), joblib.load('models/scaler_liver.pkl')),
        'kidney': (joblib.load('models/model_kidney.pkl'), joblib.load('models/scaler_kidney.pkl'))
    }

try:
    assets = load_all_assets()
except Exception as e:
    st.error("Some model files are missing. Ensure all training scripts have been run successfully.")
    st.stop()

st.title("🏥 Unified AI Health Assessment")
st.markdown("Fill out your basic information below. Advanced medical fields are auto-filled with healthy baseline values, so you only need to change what you know.")

# ==========================================
# STEP 1: MAIN PAGE INPUTS (SIMPLIFIED FOR USERS)
# ==========================================
st.header("1. Personal Profile & Vitals")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 100, 45)
    sex = st.selectbox("Biological Sex", ["Male", "Female"])
    sex_val = 1 if sex == "Male" else 0
    pregnancies = st.number_input("Pregnancies", 0, 20, 0) if sex == "Female" else 0

with col2:
    bmi = st.number_input("BMI (Body Mass Index)", 10.0, 50.0, 24.0)
    bp = st.number_input("Resting Blood Pressure (mm Hg)", 50, 200, 120)
    max_hr = st.number_input("Maximum Heart Rate", 60, 220, 150)

with col3:
    glucose = st.number_input("Glucose Level (mg/dL)", 0, 300, 90)
    fbs = 1 if glucose > 120 else 0
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 180)

st.header("2. Advanced Lab Results (Optional)")
with st.expander("Click here if you have a detailed blood or urine test report", expanded=False):
    st.info("💡 These values are pre-filled with healthy averages. Leave them as-is if you don't know them.")
    
    lab1, lab2, lab3 = st.columns(3)
    with lab1:
        st.markdown("**Metabolic & Renal**")
        insulin = st.number_input("Insulin (IU/mL)", 0, 900, 15)
        skin = st.number_input("Skin Thickness", 0, 100, 20)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.4)
        serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.1, 15.0, 0.9)
        hemoglobin = st.number_input("Hemoglobin (g/dL)", 3.0, 18.0, 14.0)
        spec_gravity = st.number_input("Specific Gravity", 1.000, 1.030, 1.020, format="%.3f")
        sugar_kidney = st.selectbox("Urine Sugar Level", [0, 1, 2, 3, 4, 5], index=0)
        albumin_kidney = st.selectbox("Urine Albumin Level", [0, 1, 2, 3, 4], index=0)
        
    with lab2:
        st.markdown("**Liver Function**")
        tot_bilirubin = st.number_input("Total Bilirubin", 0.1, 10.0, 0.8)
        dir_bilirubin = st.number_input("Direct Bilirubin", 0.1, 5.0, 0.2)
        alk_phos = st.number_input("Alkaline Phosphotase", 20, 500, 85)
        sgpt = st.number_input("SGPT (ALT)", 5, 200, 30)
        sgot = st.number_input("SGOT (AST)", 5, 200, 30)
        tot_proteins = st.number_input("Total Proteins", 2.0, 10.0, 6.8)
        albumin = st.number_input("Serum Albumin", 1.0, 6.0, 4.0)
        ag_ratio = st.number_input("A/G Ratio", 0.1, 3.0, 1.2)
        
    with lab3:
        st.markdown("**Cardiac Assessment**")
        chest_pain = st.selectbox("Chest Pain Type", [1, 2, 3, 4], index=3, help="4 means asymptomatic (Normal)")
        rest_ecg = st.selectbox("Resting ECG", [0, 1, 2], index=0)
        ex_angina = st.selectbox("Exercise Induced Angina", [0, 1], index=0)
        oldpeak = st.number_input("ST Depression", 0.0, 6.2, 0.0)
        slope = st.selectbox("ST Slope", [1, 2, 3], index=1)
        vessels = st.selectbox("Major Vessels Colored", [0, 1, 2, 3], index=0)
        thal = st.selectbox("Thalassemia", [3, 6, 7], index=0, help="3 means Normal")

# --- Feature Routing ---
diab_input = pd.DataFrame([{'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': bp, 'SkinThickness': skin, 'Insulin': insulin, 'BMI': bmi, 'DiabetesPedigreeFunction': dpf, 'Age': age}])
heart_input = pd.DataFrame([{'Age': age, 'Sex': sex_val, 'ChestPain': chest_pain, 'RestingBP': bp, 'Cholesterol': cholesterol, 'FastingBloodSugar': fbs, 'RestingECG': rest_ecg, 'MaxHeartRate': max_hr, 'ExerciseAngina': ex_angina, 'Oldpeak': oldpeak, 'Slope': slope, 'MajorVessels': vessels, 'Thal': thal}])
liver_input = pd.DataFrame([{'Age': age, 'Gender': sex_val, 'Total_Bilirubin': tot_bilirubin, 'Direct_Bilirubin': dir_bilirubin, 'Alkaline_Phosphotase': alk_phos, 'Alamine_Aminotransferase': sgpt, 'Aspartate_Aminotransferase': sgot, 'Total_Protiens': tot_proteins, 'Albumin': albumin, 'Albumin_and_Globulin_Ratio': ag_ratio}])

k_model = assets['kidney'][0]
expected_k_features = k_model.n_features_in_
kidney_dict = {'Age': age, 'Blood Pressure': bp, 'Specific Gravity': spec_gravity, 'Albumin': albumin_kidney, 'Sugar': sugar_kidney, 'Serum Creatinine': serum_creatinine, 'Hemoglobin': hemoglobin}
while len(kidney_dict) < expected_k_features:
    kidney_dict[f'Feature_{len(kidney_dict)+1}'] = 0.0
kidney_input = pd.DataFrame([kidney_dict]).iloc[:, :expected_k_features]

# ==========================================
# STEP 2: UNIFIED DIAGNOSTIC OUTPUT & DYNAMIC SHAP
# ==========================================
st.markdown("---")
st.subheader("🤖 Unified Diagnostic Engine")

if st.button("Run Full Clinical Diagnostic Suite", type="primary"):
    results = {}
    inputs_map = {'diabetes': diab_input, 'heart': heart_input, 'liver': liver_input, 'kidney': kidney_input}
    
    # 1. Run Simultaneous Inference quietly in the background
    for name, (model, scaler) in assets.items():
        inp = inputs_map[name]
        
        # Pad kidney feature count dynamically
        if name == 'kidney':
            expected_features = model.n_features_in_
            current_features = inp.shape[1]
            if current_features < expected_features:
                for i in range(current_features, expected_features):
                    inp[f'Pad_{i}'] = 0.0
            elif current_features > expected_features:
                inp = inp.iloc[:, :expected_features]

        # Use .values to bypass column name mismatches
        scaled = scaler.transform(inp.values)
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]
        
        # Store comprehensive data for dynamic SHAP rendering
        results[name] = {
            'prediction': pred, 
            'probability': prob, 
            'scaled_input': scaled, 
            'raw_input': inp
        }

    # Save to session state so UI survives button clicks/downloads
    st.session_state['results'] = results
    st.session_state['master_input'] = {'Age': age, 'BMI': bmi, 'Glucose': glucose, 'BloodPressure': bp, 'Cholesterol': cholesterol}
    st.session_state['analyzed'] = True

# --- Render the Unified Results ---
if st.session_state.get('analyzed', False):
    results = st.session_state['results']
    
    # 2. Unified Summary Sentence Logic
    flagged_diseases = [name.capitalize() for name, data in results.items() if data['prediction'] == 1]
    
    if len(flagged_diseases) > 0:
        st.error(f"⚠️ **Unified Assessment Flag:** The AI has detected potential high-risk indicators for: **{', '.join(flagged_diseases)}**.")
    else:
        st.success("✅ **Unified Assessment Clear:** No immediate high-risk indicators detected across the four major clinical modules.")
        
    # 3. Dynamic SHAP Explainability Engine
    st.markdown("### 🔍 AI Rationale (Feature Impact)")
    
    # Only explain the flagged diseases (or the highest probability one if all are clear)
    diseases_to_explain = [name for name, data in results.items() if data['prediction'] == 1]
    
    if not diseases_to_explain:
        highest_risk = max(results, key=lambda k: results[k]['probability'])
        diseases_to_explain = [highest_risk]
        st.info(f"All clear. Showing feature impact for **{highest_risk.capitalize()}** (closest to threshold at {results[highest_risk]['probability']:.1%}).")
    
    # Display charts neatly in columns based on how many diseases flagged
    cols = st.columns(2)
    saved_image_path = None
    
    for idx, disease in enumerate(diseases_to_explain):
        col_idx = idx % 2
        with cols[col_idx]:
            st.markdown(f"**{disease.capitalize()} Risk Drivers**")
            
            model = assets[disease][0]
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(results[disease]['scaled_input'])
            importances = shap_vals[1][0] if isinstance(shap_vals, list) else (shap_vals[0, :, 1] if len(shap_vals.shape) == 3 else shap_vals[0])
            
            # --- NEW: Filter out dummy/padded features from the graph ---
            clean_cols, clean_imps = [], []
            for col, imp in zip(results[disease]['raw_input'].columns, importances):
                if not str(col).startswith("Pad_") and not str(col).startswith("Feature_"):
                    clean_cols.append(col)
                    clean_imps.append(imp)
            
            fig, ax = plt.subplots(figsize=(6, 3))
            colors = ['red' if x > 0 else 'green' for x in clean_imps]
            ax.barh(clean_cols, clean_imps, color=colors)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Save the first generated plot for the PDF report
            if idx == 0:
                fig.savefig("temp_shap.png", bbox_inches='tight', dpi=150)
                saved_image_path = "temp_shap.png"

    # --- Booking & Report Center ---
    st.markdown("---")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("📅 Doctor Appointment Booking")
        with st.form("appt"):
            p_name = st.text_input("Patient Name")
            p_date = st.date_input("Preferred Date")
            spec = st.selectbox("Specialist Consultation", ["General Physician", "Endocrinologist", "Cardiologist", "Hepatologist", "Nephrologist"])
            if st.form_submit_button("Confirm Booking"):
                st.success(f"Appointment booked for {p_name} with {spec} on {p_date}.")
                
    with c2:
        st.subheader("📄 Clinical PDF Summary")
        # For the PDF, we pass a general flag if ANY disease was flagged, to keep the current PDF format working.
        is_any_high_risk = 1 if len(flagged_diseases) > 0 else 0
        highest_prob = max([data['probability'] for data in results.values()])
        
        pdf_bytes = generate_report(
            name=p_name if 'p_name' in locals() else "Patient",
            input_dict=st.session_state['master_input'],
            prediction=is_any_high_risk,
            prob=highest_prob,
            shap_image_path=saved_image_path if saved_image_path else None
        )
        st.download_button("⬇️ Download Master Diagnostic PDF", data=bytes(pdf_bytes), file_name="Unified_Multi_Disease_Report.pdf", mime="application/pdf", type="primary")