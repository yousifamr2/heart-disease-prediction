from sqlalchemy import Column, Integer, Float, String, LargeBinary, JSON
from core.database import Base

class PatientPrediction(Base):
    __tablename__ = "patients_predictions"

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer)
    sex = Column(Integer)
    chest_pain_type = Column(Integer)
    resting_bp_s = Column(Integer)
    cholesterol = Column(Integer)
    fasting_blood_sugar = Column(Integer)
    resting_ecg = Column(Integer)
    max_heart_rate = Column(Integer)
    exercise_angina = Column(Integer)
    oldpeak = Column(Float)
    ST_slope = Column(Integer)

    # ── Prediction Results ─────────────────────────────────────────────
    prediction  = Column(Integer, nullable=True)   # 0 or 1 (binary from model)
    probability = Column(Float,   nullable=True)   # 0.0 – 100.0 % from model
    risk_level  = Column(String,  nullable=True)   # "Low Risk" / "Moderate Risk" / "High Risk"
    decision    = Column(String,  nullable=True)   # "low" / "high" (data-driven @ 41%)

    # ── Report Cache ───────────────────────────────────────────────────
    shap_image      = Column(LargeBinary, nullable=True)
    llm_report_json = Column(JSON,        nullable=True)
