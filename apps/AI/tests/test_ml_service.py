import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import io

from app.services.ml_service import ml_service

def test_normalize_shap_dict():
    raw_shap = {"age": [0.15], "sex": 0.05, "invalid": "text"}
    normalized = ml_service._normalize_shap_dict(raw_shap)
    
    assert normalized["age"] == 0.15
    assert normalized["sex"] == 0.05
    assert "chest pain type" in normalized
    assert normalized["chest pain type"] == 0.1  # default

@patch("app.services.ml_service.requests.post")
def test_predict_single_success(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"prediction": 1, "probability": 85.5}
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response
    
    data = [50, 1, 0, 120, 200, 0, 0, 150, 0, 0, 1]
    result = ml_service.predict_single(data)
    
    assert result == 1
    mock_post.assert_called_once()

@patch("app.services.ml_service.requests.post")
def test_assess_full_prediction_with_api_call(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "prediction": 1, 
        "probability": 85.5,
        "shap_values": {"age": 0.2}
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response
    
    data = [50, 1, 0, 120, 200, 0, 0, 150, 0, 0, 1]
    assessment, shap_data = ml_service.assess_full_prediction(data, probability=None)
    
    assert assessment.risk_level.value == "High Risk"
    assert shap_data["age"] == 0.2
    assert "sex" in shap_data

def test_generate_shap_image():
    shap_data = {"age": 0.15, "sex": -0.05}
    image_bytes = ml_service.generate_shap_image(shap_data)
    
    assert isinstance(image_bytes, bytes)
    assert len(image_bytes) > 0
    # Check for PNG magic number
    assert image_bytes.startswith(b"\x89PNG")
