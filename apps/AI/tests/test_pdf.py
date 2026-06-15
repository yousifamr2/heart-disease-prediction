import pytest
from unittest.mock import patch

def test_pdf_fallback_internal_gateway(client, mock_headers, db):
    # This tests the branch in internal_gateway where pdf_binary is None
    # We will mock _generate_medical_pdf_bytes to return None (Playwright failure)
    
    from db.models import LabTest, Prediction, User, Lab
    
    # Create test patient
    user = User(id="user999", national_id="999", username="Test User", password="fake", email="test@test.com")
    db.add(user)
    lab = Lab(id="lab999", name="Test Lab", lab_code="TEST999", address="Test Address")
    db.add(lab)
    db.commit()

    patient = LabTest(
        id="pdf-patient", national_id="999", lab_id="lab999",
        age=60, sex=1, chest_pain_type=4, resting_bp_s=140, cholesterol=289,
        fasting_blood_sugar=0, resting_ecg=0, max_heart_rate=110,
        exercise_angina=1, oldpeak=1.5, st_slope=2
    )
    db.add(patient)
    
    pred = Prediction(id="pred999", lab_test_id="pdf-patient", prediction_result=1, prediction_percentage=90.0, risk_level="High Risk", decision="high", pdf_binary=None)
    db.add(pred)
    db.commit()

    with patch("api.endpoints.internal_gateway._generate_medical_pdf_bytes") as mock_pdf:
        mock_pdf.return_value = None
        
        response = client.post("/internal/report", json={"target_id": "pdf-patient", "user_id": "test"}, headers=mock_headers)
        
        # Should return JSON fallback instead of 503
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "fallback_data" in data
        assert data["fallback_data"]["risk_level"] == "High Risk"
