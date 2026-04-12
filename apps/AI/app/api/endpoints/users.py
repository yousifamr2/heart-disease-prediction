from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from core.database import get_db
from db.models import PatientPrediction
from schemas.patient_schema import ClientPredict

router = APIRouter(prefix="/users", tags=["Users"])

@router.post("")
def add_user(client: ClientPredict, db: Session = Depends(get_db)):
    db_patient = PatientPrediction(
        age=client.age,
        sex=client.sex.value,
        chest_pain_type=client.chest_pain_type.value,
        resting_bp_s=client.resting_bp_s,
        cholesterol=client.cholesterol,
        fasting_blood_sugar=client.fasting_blood_sugar.value,
        resting_ecg=client.resting_ecg.value,
        max_heart_rate=client.max_heart_rate,
        exercise_angina=client.exercise_angina.value,
        oldpeak=client.oldpeak,
        ST_slope=client.ST_slope.value
    )
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)

    return {
        "message": "User added successfully",
        "id": db_patient.id
    }

@router.get("/{id}")
def get_user(id: int, db: Session = Depends(get_db)):
    patient = db.query(PatientPrediction).filter(PatientPrediction.id == id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="User not found")

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
            "ST_slope": patient.ST_slope,
        },
        "prediction": patient.prediction
    }
