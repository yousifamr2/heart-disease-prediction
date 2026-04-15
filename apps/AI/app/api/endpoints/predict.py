from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from core.database import get_db
from db.models import PatientPrediction
from services.ml_service import ml_service
import pandas as pd

router = APIRouter(tags=["Prediction"])

@router.get("/predict/{id}")
def get_prediction(id: int, db: Session = Depends(get_db)):
    patient = db.query(PatientPrediction).filter(PatientPrediction.id == id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="User not found")

    # If already evaluated, return stored result
    if patient.prediction is not None:
        return {
            "id":          id,
            "prediction":  patient.prediction,
            "probability": patient.probability,
            "risk_level":  patient.risk_level,
            "decision":    patient.decision,
        }

    # If not evaluated, trigger evaluation logic (same as POST)
    return predict(id, db)


@router.post("/predict/{id}")
def predict(id: int, db: Session = Depends(get_db)):
    patient = db.query(PatientPrediction).filter(PatientPrediction.id == id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="User not found")

    # Prepare input data for the model
    data = [
        patient.age, patient.sex, patient.chest_pain_type,
        patient.resting_bp_s, patient.cholesterol, patient.fasting_blood_sugar,
        patient.resting_ecg, patient.max_heart_rate, patient.exercise_angina,
        patient.oldpeak, patient.ST_slope
    ]

    # Calculate assessment
    # If we already have a probability but missing metadata, we avoid the API call
    assessment, _ = ml_service.assess_full_prediction(data, probability=patient.probability)

    # Persist results
    patient.prediction  = 1 if assessment.decision.value == "high" else 0
    patient.probability = assessment.probability_pct
    patient.risk_level  = assessment.risk_level.value
    patient.decision    = assessment.decision.value
    db.commit()

    return {
        "id":             id,
        "prediction":     patient.prediction,
        "probability":    patient.probability,
        "risk_level":     patient.risk_level,
        "decision":       patient.decision,
        "risk_color":     assessment.risk_color,
        "decision_label": assessment.decision_label,
    }


@router.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    df.columns = df.columns.str.strip()

    missing = [col for col in ml_service.required_cols if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing columns in CSV: {missing}"
        )

    feature_df  = df[ml_service.required_cols].copy()
    predictions = ml_service.predict_dataframe(feature_df)
    df["prediction"] = predictions
    return df.to_dict(orient="records")
