"""
risk_classifier.py
──────────────────
Hybrid Decision System for Heart Disease Risk Classification.

Architecture:
  - DECISION  → binary (low / high), data-driven threshold @ 41%
                Used for actual medical logic: alerts, LLM tone, PDF urgency.

  - RISK LEVEL → 3-tier (low / moderate / high), clinically-readable
                 Used ONLY for UI display and patient communication.

Separation principle:
  Medical decisions MUST NOT depend on the UI risk level.
  The UI risk level MUST NOT drive any system behaviour.
"""

from dataclasses import dataclass
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

# Data-driven threshold (computed via Youden's J on the actual dataset, AUC=0.977)
DECISION_THRESHOLD: float = 0.41   # ← the ONLY value that drives system decisions

# UI display boundaries (clinically motivated, widened to avoid micro-zones)
UI_LOW_MAX:      float = 0.30   # < 30%        → Low Risk (UI)
UI_MODERATE_MAX: float = 0.65   # 30% – 65%   → Moderate Risk (UI)
                                 # > 65%        → High Risk (UI)


# ─────────────────────────────────────────────────────────────────────────────
#  Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class Decision(str, Enum):
    """Binary medical decision — drives alerts, recommendations, LLM tone."""
    LOW  = "low"
    HIGH = "high"


class RiskLevel(str, Enum):
    """3-tier UI label — display only, no decision logic."""
    LOW      = "Low Risk"
    MODERATE = "Moderate Risk"
    HIGH     = "High Risk"


# ─────────────────────────────────────────────────────────────────────────────
#  Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskAssessment:
    """
    Complete risk assessment returned for every prediction.

    Attributes
    ----------
    probability   : float  — raw model output (0.0 – 1.0)
    probability_pct: float — same value as a percentage (0 – 100)
    decision      : Decision   — SYSTEM logic: "low" | "high"
    risk_level    : RiskLevel  — UI label: "Low Risk" | "Moderate Risk" | "High Risk"
    decision_label: str    — human-readable decision for API / LLM
    risk_color    : str    — hex color for frontend badge
    """
    probability:     float
    probability_pct: float
    decision:        Decision
    risk_level:      RiskLevel
    decision_label:  str
    risk_color:      str

    def to_dict(self) -> dict:
        """Serialize to a plain dict, suitable for JSON API responses."""
        return {
            "probability":      round(self.probability, 4),
            "probability_pct":  round(self.probability_pct, 2),
            "decision":         self.decision.value,
            "decision_label":   self.decision_label,
            "risk_level":       self.risk_level.value,
            "risk_color":       self.risk_color,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Core functions
# ─────────────────────────────────────────────────────────────────────────────

def get_decision(probability: float) -> Decision:
    """
    Binary medical decision based on the data-driven threshold (0.41).

    Parameters
    ----------
    probability : float — model output in range [0.0, 1.0]

    Returns
    -------
    Decision.HIGH if probability >= DECISION_THRESHOLD, else Decision.LOW.

    ⚠️  This is the ONLY function allowed to drive system behaviour.
        Do NOT use get_risk_level() for medical decisions.
    """
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    return Decision.HIGH if probability >= DECISION_THRESHOLD else Decision.LOW


def get_risk_level(probability: float) -> RiskLevel:
    """
    3-tier UI label for patient-facing display.

    Boundaries (clinical, not data-driven):
      < 30%        → Low Risk
      30% – 65%    → Moderate Risk
      > 65%        → High Risk

    ⚠️  FOR DISPLAY ONLY — must not influence any system decision.

    Parameters
    ----------
    probability : float — model output in range [0.0, 1.0]

    Returns
    -------
    RiskLevel enum value.
    """
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"probability must be in [0, 1], got {probability}")

    if probability < UI_LOW_MAX:
        return RiskLevel.LOW
    elif probability <= UI_MODERATE_MAX:
        return RiskLevel.MODERATE
    else:
        return RiskLevel.HIGH


def _decision_label(decision: Decision) -> str:
    return (
        "Heart Disease Detected — Medical attention required"
        if decision == Decision.HIGH
        else "No Heart Disease Detected — Routine monitoring advised"
    )


def _risk_color(risk_level: RiskLevel) -> str:
    """Hex color for the frontend badge."""
    return {
        RiskLevel.LOW:      "#4ade80",   # green
        RiskLevel.MODERATE: "#facc15",   # yellow
        RiskLevel.HIGH:     "#f87171",   # red
    }[risk_level]


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def assess_risk(probability: float) -> RiskAssessment:
    """
    Full hybrid risk assessment — single call for both decision and display.

    Parameters
    ----------
    probability : float
        Raw model probability in range [0.0, 1.0].

    Returns
    -------
    RiskAssessment dataclass with both decision (binary) and risk_level (UI).

    Example
    -------
    >>> result = assess_risk(0.52)
    >>> result.decision        # Decision.HIGH  — drives alerts
    >>> result.risk_level      # RiskLevel.MODERATE  — shown on UI
    >>> result.to_dict()       # ready for JSON API response
    """
    decision   = get_decision(probability)
    risk_level = get_risk_level(probability)

    return RiskAssessment(
        probability      = probability,
        probability_pct  = round(probability * 100, 2),
        decision         = decision,
        risk_level       = risk_level,
        decision_label   = _decision_label(decision),
        risk_color       = _risk_color(risk_level),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Demo — run directly: python risk_classifier.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_probs = [0.10, 0.25, 0.35, 0.41, 0.52, 0.66, 0.85, 0.95]

    header = (
        f"{'Probability':>13} │ "
        f"{'Decision':^8} │ "
        f"{'UI Risk Level':^16} │ "
        f"System Decision Label"
    )
    print("\n" + "═" * 90)
    print("  HYBRID RISK ASSESSMENT — DEMO")
    print("═" * 90)
    print(f"  Data-driven threshold  : {DECISION_THRESHOLD*100:.1f}%  (Youden's J, AUC=0.977)")
    print(f"  UI boundaries          : <{UI_LOW_MAX*100:.0f}% Low  |  "
          f"{UI_LOW_MAX*100:.0f}–{UI_MODERATE_MAX*100:.0f}% Moderate  |  "
          f">{UI_MODERATE_MAX*100:.0f}% High")
    print("═" * 90)
    print(f"  {header}")
    print("  " + "─" * 88)

    for p in demo_probs:
        result = assess_risk(p)
        decision_icon   = "🔴" if result.decision == Decision.HIGH else "🟢"
        risk_icon       = {"Low Risk": "🟢", "Moderate Risk": "🟡", "High Risk": "🔴"}[
            result.risk_level.value
        ]
        print(
            f"  {result.probability_pct:>10.1f}%  │ "
            f" {decision_icon} {result.decision.value:<5}  │ "
            f" {risk_icon} {result.risk_level.value:<14} │ "
            f"  {result.decision_label}"
        )

    print("═" * 90)
    print("\n📦  Sample API response (assess_risk(0.52).to_dict()):")
    import json
    print(json.dumps(assess_risk(0.52).to_dict(), indent=4))
