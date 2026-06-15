import pytest
from app.services.risk_classifier import assess_risk, Decision, RiskLevel

def test_assess_risk_low():
    res = assess_risk(0.10)
    assert res.decision == Decision.LOW
    assert res.risk_level == RiskLevel.LOW

def test_assess_risk_moderate_low_decision():
    res = assess_risk(0.40)
    assert res.decision == Decision.LOW
    assert res.risk_level == RiskLevel.MODERATE

def test_assess_risk_moderate_high_decision():
    res = assess_risk(0.42)
    assert res.decision == Decision.HIGH
    assert res.risk_level == RiskLevel.MODERATE

def test_assess_risk_high():
    res = assess_risk(0.70)
    assert res.decision == Decision.HIGH
    assert res.risk_level == RiskLevel.HIGH

def test_assess_risk_invalid():
    with pytest.raises(ValueError):
        assess_risk(1.5)
