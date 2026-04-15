"""
api/endpoints/report.py
──────────────────────────────────────────────────────────────────────────
Report Endpoint — pure orchestrator, zero business logic.

Pipeline:
  1. Load patient data from DB
  2. assess_full_prediction() → RiskAssessment + shap_data
  3. chart_service → base64 PNG charts (cached)
  4. llm.generate_report() → explanation + recommendations
  5. renderer.render_report(context) → HTML string
  6. pdf_exporter.html_to_pdf(html) → PDF bytes
  7. Return PDF as HTTP Response
"""

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from core.database import get_db
from db.models import PatientPrediction
from services.ml_service import ml_service
from services import chart_service
from services.report import renderer, pdf_exporter

import sys
from pathlib import Path

# ── LLM import (lives outside /app) ─────────────────────────────────────────
AI_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(AI_DIR) not in sys.path:
    sys.path.append(str(AI_DIR))

try:
    from LLM.llm import HeartDiseaseConsultant
    consultant = HeartDiseaseConsultant()
except Exception as e:
    print("Warning: Could not initialize HeartDiseaseConsultant:", e)
    consultant = None

router = APIRouter(tags=["Report"])


@router.get("/predict/{id}/report")
def get_prediction_report(id: int, db: Session = Depends(get_db)):
    # ── 1. Load patient ───────────────────────────────────────────────────
    patient = db.query(PatientPrediction).filter(PatientPrediction.id == id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="User not found")
    if patient.prediction is None:
        raise HTTPException(
            status_code=400,
            detail="Patient has not been evaluated yet. Call POST /predict/{id} first."
        )
    if not consultant:
        raise HTTPException(status_code=500, detail="LLM Consultant is not initialized.")

    # ── 2. Prediction + Risk Assessment ──────────────────────────────────
    data = [
        patient.age, patient.sex, patient.chest_pain_type,
        patient.resting_bp_s, patient.cholesterol, patient.fasting_blood_sugar,
        patient.resting_ecg, patient.max_heart_rate, patient.exercise_angina,
        patient.oldpeak, patient.ST_slope,
    ]
    assessment, shap_data = ml_service.assess_full_prediction(data)

    # ── 3. Charts (lru_cache — same SHAP = instant return) ───────────────
    shap_tuple    = tuple(sorted(shap_data.items()))
    feat_chart    = chart_service.generate_feature_importance_chart(shap_tuple)
    shap_chart    = chart_service.generate_shap_waterfall_chart(shap_tuple)

    # ── 4. Patient Info (English labels) ─────────────────────────────────
    patient_info = {
        "Age":                    patient.age,
        "Sex":                    "Male" if patient.sex == 1 else "Female",
        "Chest Pain Type":        patient.chest_pain_type,
        "Resting Blood Pressure": patient.resting_bp_s,
        "Cholesterol":            patient.cholesterol,
        "Fasting Blood Sugar":    patient.fasting_blood_sugar,
        "Resting ECG":            patient.resting_ecg,
        "Max Heart Rate":         patient.max_heart_rate,
        "Exercise Angina":        "Yes" if patient.exercise_angina == 1 else "No",
        "Oldpeak":                patient.oldpeak,
        "ST Slope":               patient.ST_slope,
    }

    # ── 5. LLM (use cache if report exists and is valid) ─────────────────
    top_features = sorted(shap_data.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    if (patient.llm_report_json
            and "explanation" in patient.llm_report_json
            and "error" not in patient.llm_report_json):
        llm_result = patient.llm_report_json
    else:
        llm_result = consultant.generate_report(
            probability   = assessment.probability_pct,
            decision      = assessment.decision.value,
            ui_risk_level = assessment.risk_level.value,
            top_features  = top_features,
        )
        # Cache in DB
        patient.llm_report_json = llm_result
        patient.probability     = assessment.probability_pct
        patient.risk_level      = assessment.risk_level.value
        patient.decision        = assessment.decision.value
        db.commit()

    # ── 6. Build HTML context ─────────────────────────────────────────────
    # font_path needed by xhtml2pdf to load the Amiri Arabic font
    assets_dir = Path(__file__).resolve().parent.parent.parent.parent / "app" / "assets" / "fonts" / ""
    context = {
        "patient_info":    patient_info,
        "probability":     round(assessment.probability_pct, 1),
        "decision":        assessment.decision.value,           # "low" | "high"
        "ui_risk_level":   assessment.risk_level.value,        # display badge
        "ui_risk_color":   assessment.risk_color,              # hex
        "feat_chart_b64":  feat_chart,
        "shap_chart_b64":  shap_chart,
        "explanation":     llm_result.get("explanation", ""),
        "recommendations": llm_result.get("recommendations", []),
        "generated_at":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "font_path":       str(assets_dir).replace("\\", "/") + "/",
    }

    # ── 7. Render HTML → PDF ──────────────────────────────────────────────
    from services.report import renderer as _renderer
    from services.report import pdf_exporter as _pdf_exporter

    html      = _renderer.render_report(context)
    pdf_bytes = _pdf_exporter.html_to_pdf(html)

    return Response(
        content=bytes(pdf_bytes),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=artemis_report_patient_{id}.pdf"
        },
    )
