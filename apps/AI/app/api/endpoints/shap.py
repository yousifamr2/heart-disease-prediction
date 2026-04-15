from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from core.database import get_db
from db.models import PatientPrediction
from services.ml_service import ml_service
import io

router = APIRouter(prefix="/shap", tags=["Explainability"])

@router.get("/{id}")
def show_shap(id: int, db: Session = Depends(get_db)):
    patient = db.query(PatientPrediction).filter(PatientPrediction.id == id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="User not found")

    if patient.shap_image:
        return StreamingResponse(io.BytesIO(patient.shap_image), media_type="image/png")

    data = [
        patient.age, patient.sex, patient.chest_pain_type,
        patient.resting_bp_s, patient.cholesterol, patient.fasting_blood_sugar,
        patient.resting_ecg, patient.max_heart_rate, patient.exercise_angina,
        patient.oldpeak, patient.ST_slope
    ]

    risk_score, shap_data = ml_service.get_risk_and_shap(data)
    image_bytes = ml_service.generate_shap_image(shap_data)

    patient.shap_image = image_bytes
    db.commit()

    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")
