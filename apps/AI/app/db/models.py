from sqlalchemy import Column, Integer, Float, LargeBinary
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
    
    prediction = Column(Integer, nullable=True)
    shap_image = Column(LargeBinary, nullable=True)
