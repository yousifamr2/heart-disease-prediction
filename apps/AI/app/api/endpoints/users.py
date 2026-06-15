from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db.database import get_db
from db.models import LabTest, Prediction

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/{id}")
def get_user(id: str, db: Session = Depends(get_db)):
    patient = db.query(LabTest).filter(LabTest.id == id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="LabTest not found")

    prediction_record = db.query(Prediction).filter(Prediction.lab_test_id == id).first()
    
    return {
        "id": patient.id,
        "data": {
            "age": patient.age,
            "sex": patient.sex,
            "chest_pain_type": patient.chest_pain_type,
            "resting_bp_s": patient.resting_bp_s,
            "cholesterol": patient.cholesterol,
            "fasting_blood_sugar": patient.fasting_blood_sugar,
            "resting_ecg": patient.resting_ecg,
            "max_heart_rate": patient.max_heart_rate,
            "exercise_angina": patient.exercise_angina,
            "oldpeak": patient.oldpeak,
            "ST_slope": patient.st_slope,
        },
        "prediction": prediction_record.prediction_result if prediction_record else None
    }

