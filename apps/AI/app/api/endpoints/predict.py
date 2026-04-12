from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from core.database import get_db
from db.models import PatientPrediction
from services.ml_service import ml_service
import pandas as pd

router = APIRouter(tags=["Prediction"])

@router.post("/predict/{id}")
def predict(id: int, db: Session = Depends(get_db)):
    patient = db.query(PatientPrediction).filter(PatientPrediction.id == id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="User not found")
    if patient.prediction is not None:
        raise HTTPException(status_code=400, detail="User already evaluated")

    data = [
        patient.age, patient.sex, patient.chest_pain_type,
        patient.resting_bp_s, patient.cholesterol, patient.fasting_blood_sugar,
        patient.resting_ecg, patient.max_heart_rate, patient.exercise_angina,
        patient.oldpeak, patient.ST_slope
    ]

    prediction = ml_service.predict_single(data)
    
    patient.prediction = prediction
    db.commit()

    return {
        "id": id,
        "prediction": patient.prediction
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

    feature_df = df[ml_service.required_cols].copy()
    predictions = ml_service.predict_dataframe(feature_df)
    
    df["prediction"] = predictions
    return df.to_dict(orient="records")
