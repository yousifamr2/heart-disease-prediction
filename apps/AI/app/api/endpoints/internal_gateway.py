"""
Internal AI routes — callable ONLY by the Node.js gateway with X-INTERNAL-API-KEY.

Security decisions:
- No national_id fallback on internal routes (strict lab_tests.id only) to reduce IDOR surface.
- Public /predict, /shap, /report are removed from the app router; use these POST endpoints only.
"""

from __future__ import annotations

import io
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.orm import Session

from core.security import verify_internal_api_key
from db.database import get_db
from db.models import Lab, LabTest, Prediction, User
from schemas.internal import InternalTargetRequest
from services import chart_service
from services.ml_service import ml_service
from services.pdf_service import generate_medical_report_pdf

AI_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(AI_DIR) not in sys.path:
    sys.path.append(str(AI_DIR))

try:
    from app.services.llm_service import HeartDiseaseConsultant

    consultant = HeartDiseaseConsultant()
except Exception as e:
    print("Warning: Could not initialize HeartDiseaseConsultant:", e)
    consultant = None

router = APIRouter(
    prefix="/internal",
    tags=["Internal AI"],
    dependencies=[Depends(verify_internal_api_key)],
)


def _lab_test_by_id(db: Session, lab_test_id: str) -> LabTest:
    patient = db.query(LabTest).filter(LabTest.id == lab_test_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="LabTest not found")
    return patient


def _apply_user_id(prediction_record: Prediction, user_id: str | None, db: Session) -> None:
    if user_id and prediction_record.user_id != user_id:
        prediction_record.user_id = user_id
        db.commit()


def _generate_medical_pdf_bytes(
    patient: LabTest,
    prediction_record: Prediction,
    patient_name: str,
    lab_record: Lab | None,
    shap_data: dict,
    llm_result: dict,
) -> bytes | None:
    """Charts + HTML→PDF. Returns None on failure (logged)."""
    try:
        shap_tuple = tuple(sorted(shap_data.items()))
        feat_chart = chart_service.generate_feature_importance_chart(shap_tuple)
        shap_chart = chart_service.generate_shap_waterfall_chart(shap_tuple)

        patient_data = {
            "name": patient_name,
            "gender": "Male" if patient.sex == 1 else "Female",
            "dob": "N/A",
            "national_id": patient.national_id or "N/A",
            "address": "N/A",
            "age": patient.age,
            "cp": patient.chest_pain_type,
            "trestbps": patient.resting_bp_s,
            "chol": patient.cholesterol,
            "fbs": patient.fasting_blood_sugar,
            "restecg": patient.resting_ecg,
            "thalach": patient.max_heart_rate,
            "exang": "Yes" if patient.exercise_angina == 1 else "No",
            "oldpeak": patient.oldpeak,
            "slope": patient.st_slope,
        }

        risk_score = (
            round(prediction_record.prediction_percentage, 1)
            if prediction_record.prediction_percentage
            else 0.0
        )
        llm_report = {
            "summary": llm_result.get("explanation", ""),
            "recommendations": llm_result.get("recommendations", []),
        }
        images_base64 = {
            "university_logo": "",
            "risk_gauge": feat_chart,
            "shap_plot": shap_chart,
        }
        lab_data = {
            "name": lab_record.name if lab_record else "N/A",
            "address": lab_record.address if lab_record else "N/A",
        }
        lab_test_data = {"id": patient.id}

        pdf_bytes_io = generate_medical_report_pdf(
            patient_data=patient_data,
            risk_score=risk_score,
            llm_report=llm_report,
            images_base64=images_base64,
            lab_data=lab_data,
            lab_test_data=lab_test_data,
        )
        return pdf_bytes_io.getvalue()
    except Exception as e:
        print(f"Warning: PDF report generation failed: {e}")
        return None


@router.post("/predict")
def internal_predict(body: InternalTargetRequest, db: Session = Depends(get_db)):
    patient = _lab_test_by_id(db, body.target_id)

    user_record = db.query(User).filter(User.national_id == patient.national_id).first()
    patient_name = user_record.username if user_record else "Anonymous"
    lab_record = db.query(Lab).filter(Lab.id == patient.lab_id).first()

    prediction_record = db.query(Prediction).filter(Prediction.lab_test_id == patient.id).first()
    if prediction_record and prediction_record.prediction_result is not None:
        _apply_user_id(prediction_record, body.user_id, db)
        assessment, _ = ml_service.assess_full_prediction(
            [], probability=prediction_record.prediction_percentage
        )
        return {
            "id": prediction_record.id,
            "lab_test_id": prediction_record.lab_test_id,
            "prediction": prediction_record.prediction_result,
            "probability": prediction_record.prediction_percentage,
            "risk_level": prediction_record.risk_level,
            "decision": prediction_record.decision,
            "risk_color": assessment.risk_color,
            "decision_label": assessment.decision_label,
        }

    data = [
        patient.age,
        patient.sex,
        patient.chest_pain_type,
        patient.resting_bp_s,
        patient.cholesterol,
        patient.fasting_blood_sugar,
        patient.resting_ecg,
        patient.max_heart_rate,
        patient.exercise_angina,
        patient.oldpeak,
        patient.st_slope,
    ]

    try:
        assessment, shap_data = ml_service.assess_full_prediction(data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Model inference failed: {str(e)}")

    prediction_record = db.query(Prediction).filter(Prediction.lab_test_id == patient.id).first()
    if not prediction_record:
        prediction_record = Prediction(
            id=str(uuid.uuid4()),
            lab_test_id=patient.id,
            user_id=body.user_id,
        )
        db.add(prediction_record)
    elif body.user_id:
        prediction_record.user_id = body.user_id

    prediction_record.prediction_result = 1 if assessment.decision.value == "high" else 0
    prediction_record.prediction_percentage = assessment.probability_pct
    prediction_record.risk_level = assessment.risk_level.value
    prediction_record.decision = assessment.decision.value
    prediction_record.shap_values_json = shap_data

    if assessment.decision.value == "high":
        image_bytes = ml_service.generate_shap_image(shap_data)
        prediction_record.shap_image = image_bytes

        if consultant:
            top_features = sorted(shap_data.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            try:
                patient_data = {
                    "Age": patient.age,
                    "Sex": "Male" if patient.sex == 1 else "Female",
                    "Chest Pain Type": patient.chest_pain_type,
                    "Resting BP (mm Hg)": patient.resting_bp_s,
                    "Cholesterol (mg/dl)": patient.cholesterol,
                    "Fasting Blood Sugar": patient.fasting_blood_sugar,
                    "Resting ECG": patient.resting_ecg,
                    "Max Heart Rate": patient.max_heart_rate,
                    "Exercise Angina": "Yes" if patient.exercise_angina == 1 else "No",
                    "Oldpeak": patient.oldpeak,
                    "ST Slope": patient.st_slope,
                }
                llm_result = consultant.generate_report(
                    probability=assessment.probability_pct,
                    decision=assessment.decision.value,
                    ui_risk_level=assessment.risk_level.value,
                    top_features=top_features,
                    patient_data=patient_data,
                )
                prediction_record.llm_report_json = llm_result
            except Exception as e:
                print(f"Warning: Failed to communicate with AI provider: {str(e)}")
                llm_result = {"explanation": "LLM generation failed.", "recommendations": []}
                prediction_record.llm_report_json = llm_result
        else:
            llm_result = {"explanation": "LLM Consultant is not initialized.", "recommendations": []}
            prediction_record.llm_report_json = llm_result

        pdf_bytes = _generate_medical_pdf_bytes(
            patient, prediction_record, patient_name, lab_record, shap_data, llm_result
        )
        if pdf_bytes:
            prediction_record.pdf_binary = pdf_bytes
            prediction_record.report_generated_at = datetime.utcnow().isoformat()
        else:
            prediction_record.pdf_binary = None
            prediction_record.report_generated_at = None
    else:
        prediction_record.shap_image = None
        prediction_record.llm_report_json = None
        prediction_record.pdf_binary = None
        prediction_record.report_generated_at = None

    db.commit()

    return {
        "id": prediction_record.id,
        "lab_test_id": prediction_record.lab_test_id,
        "prediction": prediction_record.prediction_result,
        "probability": prediction_record.prediction_percentage,
        "risk_level": prediction_record.risk_level,
        "decision": prediction_record.decision,
        "risk_color": assessment.risk_color,
        "decision_label": assessment.decision_label,
    }


@router.post("/shap")
def internal_shap_png(body: InternalTargetRequest, db: Session = Depends(get_db)):
    patient = _lab_test_by_id(db, body.target_id)
    prediction_record = db.query(Prediction).filter(Prediction.lab_test_id == patient.id).first()
    if not prediction_record:
        raise HTTPException(
            status_code=400,
            detail="Prediction not evaluated yet. Call POST /internal/predict first.",
        )
    if prediction_record.decision == "low":
        raise HTTPException(
            status_code=400,
            detail="SHAP image is not available for low risk predictions.",
        )

    if not prediction_record.shap_image:
        data = [
            patient.age,
            patient.sex,
            patient.chest_pain_type,
            patient.resting_bp_s,
            patient.cholesterol,
            patient.fasting_blood_sugar,
            patient.resting_ecg,
            patient.max_heart_rate,
            patient.exercise_angina,
            patient.oldpeak,
            patient.st_slope,
        ]
        _, shap_data = ml_service.assess_full_prediction(data)
        image_bytes = ml_service.generate_shap_image(shap_data)
        prediction_record.shap_image = image_bytes
        db.commit()

    return StreamingResponse(
        io.BytesIO(prediction_record.shap_image),
        media_type="image/png",
    )


@router.post("/shap/data")
def internal_shap_data(body: InternalTargetRequest, db: Session = Depends(get_db)):
    """Structured SHAP JSON for gateway/frontend (optional)."""
    patient = _lab_test_by_id(db, body.target_id)
    prediction_record = db.query(Prediction).filter(Prediction.lab_test_id == patient.id).first()
    if not prediction_record:
        raise HTTPException(
            status_code=400,
            detail="Prediction not evaluated yet. Call POST /internal/predict first.",
        )
    if prediction_record.decision == "low":
        raise HTTPException(
            status_code=400,
            detail="SHAP data is not available for low risk predictions.",
        )

    if prediction_record.shap_values_json:
        shap_data = prediction_record.shap_values_json
    else:
        data = [
            patient.age,
            patient.sex,
            patient.chest_pain_type,
            patient.resting_bp_s,
            patient.cholesterol,
            patient.fasting_blood_sugar,
            patient.resting_ecg,
            patient.max_heart_rate,
            patient.exercise_angina,
            patient.oldpeak,
            patient.st_slope,
        ]
        _, shap_data = ml_service.assess_full_prediction(data)
        prediction_record.shap_values_json = shap_data
        db.commit()

    sorted_features = sorted(shap_data.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = []
    labels = []
    values = []
    for feature_name, impact in sorted_features:
        attr_name = feature_name.replace(" ", "_")
        raw_val = getattr(patient, attr_name, "N/A")
        direction = "increase" if impact > 0 else "decrease"
        top_features.append(
            {
                "feature": feature_name,
                "value": raw_val,
                "impact": round(impact, 4),
                "direction": direction,
            }
        )
        labels.append(feature_name)
        values.append(round(impact, 4))

    top_f = top_features[0]
    direction_verb = "increased" if top_f["direction"] == "increase" else "decreased"
    readable_explanation = (
        f"The value of {top_f['feature']} ({top_f['value']}) strongly {direction_verb} "
        "the predicted heart disease risk."
    )

    return {
        "prediction_probability": prediction_record.prediction_percentage,
        "risk_level": prediction_record.risk_level,
        "top_features": top_features,
        "chart_data": {"labels": labels, "values": values},
        "explanation": readable_explanation,
    }


@router.post("/report")
def internal_report_pdf(body: InternalTargetRequest, db: Session = Depends(get_db)):
    prediction_record = db.query(Prediction).filter(Prediction.lab_test_id == body.target_id).first()
    if not prediction_record:
        raise HTTPException(
            status_code=400,
            detail="Prediction has not been evaluated yet. Call POST /internal/predict first.",
        )
    if prediction_record.decision == "low":
        raise HTTPException(
            status_code=400,
            detail="Report PDF is not available for low risk predictions.",
        )
    if not prediction_record.pdf_binary:
        patient = _lab_test_by_id(db, body.target_id)
        user_record = db.query(User).filter(User.national_id == patient.national_id).first()
        patient_name = user_record.username if user_record else "Anonymous"
        lab_record = db.query(Lab).filter(Lab.id == patient.lab_id).first()
        raw_shap = prediction_record.shap_values_json
        if raw_shap:
            shap_data = raw_shap if isinstance(raw_shap, dict) else dict(raw_shap)
        else:
            data = [
                patient.age,
                patient.sex,
                patient.chest_pain_type,
                patient.resting_bp_s,
                patient.cholesterol,
                patient.fasting_blood_sugar,
                patient.resting_ecg,
                patient.max_heart_rate,
                patient.exercise_angina,
                patient.oldpeak,
                patient.st_slope,
            ]
            _, shap_data = ml_service.assess_full_prediction(data)
        raw_llm = prediction_record.llm_report_json
        llm_result = (
            raw_llm
            if isinstance(raw_llm, dict)
            else {"explanation": "Report data unavailable.", "recommendations": []}
        )
        pdf_bytes = _generate_medical_pdf_bytes(
            patient, prediction_record, patient_name, lab_record, shap_data, llm_result
        )
        if pdf_bytes:
            prediction_record.pdf_binary = pdf_bytes
            prediction_record.report_generated_at = datetime.utcnow().isoformat()
            db.commit()

    if not prediction_record.pdf_binary:
        return {
            "success": False,
            "message": "PDF report generation failed (Playwright missing).",
            "fallback_data": {
                "prediction_probability": prediction_record.prediction_percentage,
                "risk_level": prediction_record.risk_level,
                "decision": prediction_record.decision,
            }
        }
        
    return Response(
        content=prediction_record.pdf_binary,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=artemis_report_labtest_{body.target_id}.pdf"
        },
    )


@router.post("/predict-csv")
async def internal_predict_csv(file: UploadFile = File(...)):
    """Batch CSV scoring — internal only (same as legacy /predict-csv)."""
    df = pd.read_csv(file.file)
    df.columns = df.columns.str.strip()
    missing = [col for col in ml_service.required_cols if col not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing columns in CSV: {missing}")
    feature_df = df[ml_service.required_cols].copy()
    predictions = ml_service.predict_dataframe(feature_df)
    df["prediction"] = predictions
    return df.to_dict(orient="records")


@router.get("/labtest/{lab_test_id}/summary")
def internal_labtest_summary(lab_test_id: str, db: Session = Depends(get_db)):
    """Legacy user-style payload; internal only."""
    patient = _lab_test_by_id(db, lab_test_id)
    prediction_record = db.query(Prediction).filter(Prediction.lab_test_id == lab_test_id).first()
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
        "prediction": prediction_record.prediction_result if prediction_record else None,
    }
