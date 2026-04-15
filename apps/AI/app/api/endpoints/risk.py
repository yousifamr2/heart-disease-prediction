from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from core.database import get_db
from db.models import PatientPrediction
from services.ml_service import ml_service
import pandas as pd

router = APIRouter(tags=["Risk"])

@router.get("/risk/{id}")
def get_risk(id: int, db: Session = Depends(get_db)):
    patient = db.query(PatientPrediction).filter(PatientPrediction.id == id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="User not found")
    if patient.risk_level is None:
        raise HTTPException(status_code=400, detail="User has not been evaluated yet")
    return {
        "id":          id,
        "risk_level":  patient.risk_level,
        "decision":    patient.decision,
    }
