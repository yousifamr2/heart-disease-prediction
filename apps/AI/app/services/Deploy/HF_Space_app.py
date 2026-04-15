from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import shap
from huggingface_hub import hf_hub_download

app = FastAPI(
    title="Artemis Heart Disease Prediction API",
    description="AI-powered heart disease risk assessment with SHAP explainability.",
    version="1.0.0",
)

# ── Model Loading (runs once at startup) ─────────────────────────────────────
REPO_ID  = "Omarbm52/Heart-Disease-Prediction-Artemis"
FILENAME = "best_model.pkl"

model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
model      = joblib.load(model_path)

# TreeExplainer is faster and more accurate for tree-based models (RandomForest etc.)
# Using check_additivity=False to suppress harmless warnings
explainer = shap.TreeExplainer(model)

FEATURE_COLS = [
    "age", "sex", "chest pain type", "resting bp s", "cholesterol",
    "fasting blood sugar", "resting ecg", "max heart rate",
    "exercise angina", "oldpeak", "ST slope"
]


# ── Input Schema ─────────────────────────────────────────────────────────────
class PatientData(BaseModel):
    age:                  float = Field(alias="age")
    sex:                  int   = Field(alias="sex")
    chest_pain_type:      int   = Field(alias="chest pain type")
    resting_bp_s:         float = Field(alias="resting bp s")
    cholesterol:          float = Field(alias="cholesterol")
    fasting_blood_sugar:  int   = Field(alias="fasting blood sugar")
    resting_ecg:          int   = Field(alias="resting ecg")
    max_heart_rate:       float = Field(alias="max heart rate")
    exercise_angina:      int   = Field(alias="exercise angina")
    oldpeak:              float = Field(alias="oldpeak")
    st_slope:             int   = Field(alias="ST slope")

    class Config:
        populate_by_name = True


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "Artemis Heart Disease API is running!", "status": "ok"}


@app.get("/health")
def health():
    """Health check endpoint required by Hugging Face Spaces."""
    return {"status": "healthy", "model": "RandomForestClassifier", "features": 11}


@app.post("/predict")
def predict(data: PatientData):
    """
    Predict heart disease risk and return SHAP feature importance.

    Returns:
        prediction  : int   — 0 (no disease) or 1 (disease)
        probability : float — probability % of heart disease (0–100)
        shap_values : dict  — feature name → SHAP contribution value
    """
    try:
        # ── Build input DataFrame (column names must match training data) ──
        input_df = pd.DataFrame([data.model_dump(by_alias=True)])
        input_df = input_df[FEATURE_COLS]  # enforce correct column order

        # ── Prediction ────────────────────────────────────────────────────
        prediction = int(model.predict(input_df)[0])

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_df)[0][1])
        else:
            probability = float(prediction)

        # ── SHAP Explainability ───────────────────────────────────────────
        shap_values_raw = explainer.shap_values(input_df)

        # For binary classifiers: shap_values is a list [class_0, class_1]
        # We want class 1 (heart disease) values
        if isinstance(shap_values_raw, list):
            values = shap_values_raw[1][0]          # class 1, first sample
        elif len(shap_values_raw.shape) == 3:
            values = shap_values_raw[0, :, 1]       # (samples, features, classes)
        else:
            values = shap_values_raw[0]             # (samples, features)

        shap_dict = {
            col: round(float(values[i]), 6)
            for i, col in enumerate(FEATURE_COLS)
        }

        return {
            "prediction":  prediction,
            "probability": round(probability * 100, 2),   # 0–100 scale
            "shap_values": shap_dict,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
