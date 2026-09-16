# rules.md

## Development Guardrails & AI Instructions

**1. Zero Feature Creep**
*   **Rule:** Do not suggest or implement any features outside the scope defined in `phases.md`. 
*   **Enforcement:** No user authentication, no external database connections (SQL/MongoDB), no complex frontend frameworks. Stick strictly to Streamlit, Scikit-Learn, and FPDF.

**2. Anti-Hallucination Protocol**
*   **Rule:** Every variable name, file path, and dataset reference must match `memory.md` exactly.
*   **Enforcement:** Before generating new code, verify the current directory structure. Always assume the root directory is `/AI Disease Prediction`. 

**3. MVP Data Constraints**
*   **Rule:** Use only standard, pre-cleaned datasets for the MVP (e.g., PIMA Indians Diabetes).
*   **Enforcement:** Do not write complex data cleaning pipelines. If a dataset requires more than basic imputation or scaling, find a cleaner dataset. Time is the primary constraint.

**4. Explanability Requirement**
*   **Rule:** Every prediction must be accompanied by a SHAP value visualization.
*   **Enforcement:** Do not output raw probabilities without the SHAP plot. The UI must clearly explain the *why* behind the prediction.

**5. Model Serialization**
*   **Rule:** Models must be trained in separate scripts (e.g., inside a `/scripts` or `/notebooks` folder) and saved as `.pkl` files.
*   **Enforcement:** `app.py` must only *load* models, never train them. This ensures the Streamlit app boots instantly.