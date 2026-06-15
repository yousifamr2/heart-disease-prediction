import pytest
from unittest.mock import patch, MagicMock
from app.services.llm_service import HeartDiseaseConsultant, EcgConsultant, sanitize_llm_output, _UNSAFE_PATTERNS

def test_sanitize_llm_output_unsafe_patterns():
    # Append the specific pattern the user wants to test if it's not already there
    custom_patterns = _UNSAFE_PATTERNS + [r"\btake Aspirin\b"]
    
    with patch("app.services.llm_service._UNSAFE_PATTERNS", custom_patterns):
        unsafe_text = "You should definitely take Aspirin because you have heart disease."
        sanitized = sanitize_llm_output(unsafe_text)
        
        assert "[medically reviewed]" in sanitized
        assert "take Aspirin" not in sanitized
        assert "you have heart disease" not in sanitized


@patch("app.services.llm_service.ChatGroq")
def test_heart_disease_consultant_success(mock_chat_groq):
    mock_llm_instance = MagicMock()
    mock_chat_groq.return_value = mock_llm_instance
    
    fake_json_response = {
        "explanation": "The patient features may suggest an increased risk.",
        "recommendations": ["Consult a doctor", "Exercise regularly"]
    }
    
    consultant = HeartDiseaseConsultant()
    consultant._chain = MagicMock()
    consultant._chain.invoke.return_value = fake_json_response
    
    result = consultant.generate_report(
        probability=85.5,
        decision="high",
        ui_risk_level="High Risk",
        top_features=[("age", 0.15)]
    )
    
    assert result["explanation"] == "The patient features may suggest an increased risk."
    assert "Consult a doctor" in result["recommendations"]


@patch("app.services.llm_service.ChatGroq")
def test_heart_disease_consultant_timeout_fallback(mock_chat_groq):
    mock_llm_instance = MagicMock()
    mock_chat_groq.return_value = mock_llm_instance
    
    consultant = HeartDiseaseConsultant()
    consultant._chain = MagicMock()
    consultant._chain.invoke.side_effect = TimeoutError("API timed out")
    
    result = consultant.generate_report(
        probability=85.5,
        decision="high",
        ui_risk_level="High Risk",
        top_features=[("age", 0.15)]
    )
    
    assert "Could not generate explanation" in result["explanation"]
    assert "API timed out" in result["explanation"]
    assert "Please consult your physician" in result["recommendations"][0]


@patch("app.services.llm_service.ChatGroq")
def test_ecg_consultant_timeout_fallback(mock_chat_groq):
    mock_llm_instance = MagicMock()
    mock_chat_groq.return_value = mock_llm_instance
    
    consultant = EcgConsultant()
    consultant._chain = MagicMock()
    consultant._chain.invoke.side_effect = TimeoutError("API timed out")
    
    result = consultant.generate_ecg_report(
        top_5=[{"label": "Normal ECG (NORM)", "probability": 99.0}],
        kb_context="Some context"
    )
    
    assert "We could not generate an extended narrative interpretation" in result["interpretation"]
    assert "API timed out" in result["interpretation"]
    assert result["urgency"] == "If you have chest pain, fainting, or severe shortness of breath, seek emergency care."
