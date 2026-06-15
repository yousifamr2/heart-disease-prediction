import pytest
from unittest.mock import patch
from db.models import LabTest, User, Lab

@pytest.fixture(scope="module")
def sample_patient(db):
    user = User(id="user1", national_id="12345678901234", username="Test User", password="fake", email="test@test.com")
    db.add(user)
    lab = Lab(id="lab1", name="Test Lab", lab_code="TEST001", address="Test Address")
    db.add(lab)
    db.commit()

    patient = LabTest(
        id="test-patient-id",
        national_id="12345678901234",
        lab_id="lab1",
        age=60,
        sex=1,
        chest_pain_type=4,
        resting_bp_s=140,
        cholesterol=289,
        fasting_blood_sugar=0,
        resting_ecg=0,
        max_heart_rate=110,
        exercise_angina=1,
        oldpeak=1.5,
        st_slope=2
    )
    db.add(patient)
    db.commit()
    return patient

@patch("services.ml_service.MLService._call_api")
def test_predict_hf_success(mock_api, client, mock_headers, sample_patient):
    mock_api.return_value = {
        "prediction": 1,
        "probability": 85.0,
        "shap_values": {"age": 0.5, "sex": 0.2}
    }
    
    response = client.post("/internal/predict", json={"target_id": "test-patient-id", "user_id": "user1"}, headers=mock_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["probability"] == 85.0
    assert data["risk_level"] == "High Risk"

@patch("services.ml_service.MLService._call_api")
def test_predict_hf_timeout_fallback(mock_api, client, mock_headers, sample_patient):
    # Simulate HuggingFace failure
    mock_api.side_effect = Exception("HuggingFace Timeout")
    
    # It should fallback to local ML service automatically
    response = client.post("/internal/predict", json={"target_id": "test-patient-id", "user_id": "user1"}, headers=mock_headers)
    assert response.status_code == 200
    data = response.json()
    assert "probability" in data
    assert "risk_level" in data
    # Decision label should be populated based on the local fallback probability
    
def test_risk_classifier_brackets():
    from app.services.risk_classifier import assess_risk
    
    low = assess_risk(0.39)
    assert low.risk_level.value == "Low Risk"
    
    low_boundary = assess_risk(0.40)
    assert low_boundary.risk_level.value == "Low Risk"
    
    moderate = assess_risk(0.41)
    assert moderate.risk_level.value == "Moderate Risk"
    
    moderate_high = assess_risk(0.60)
    assert moderate_high.risk_level.value == "Moderate Risk"
    
    high = assess_risk(0.61)
    assert high.risk_level.value == "High Risk"
