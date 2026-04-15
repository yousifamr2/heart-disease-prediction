"""
LLM/llm.py
──────────────────────────────────────────────────────────────────────────
LLM Layer — generates English medical explanation and recommendations.

Responsibilities:
  - Build a dynamic, non-hardcoded English prompt (build_prompt)
  - Call the LLM via LangChain
  - Sanitize output to remove unsafe absolute medical claims (sanitize_llm_output)
"""

import os
import re
from typing import Dict, List

from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv(find_dotenv())


# ─────────────────────────────────────────────────────────────────────────────
#  Output Schema — lean, template handles the rest
# ─────────────────────────────────────────────────────────────────────────────

class MedicalReport(BaseModel):
    explanation:     str        = Field(description="2-3 sentence medical explanation in English")
    recommendations: List[str]  = Field(description="3-5 specific, actionable health recommendations in English")


# ─────────────────────────────────────────────────────────────────────────────
#  Medical Safety Layer
# ─────────────────────────────────────────────────────────────────────────────

_UNSAFE_PATTERNS = [
    r"\byou have heart disease\b",
    r"\byou are (definitely|certainly|diagnosed)\b",
    r"\bthis is a (diagnosis|confirmed)\b",
    r"\byou will (die|suffer|have a heart attack)\b",
    r"\b100%\s*(certain|sure|confirmed)\b",
    r"\bdefinitely (have|has|diagnosed)\b",
    r"\bclinically confirmed\b",
    r"\byou are dying\b",
]


def sanitize_llm_output(text: str) -> str:
    """
    Remove or replace medically unsafe absolute claims from LLM output.

    Production safety layer — ALWAYS applied before returning LLM text.

    Examples of blocked phrases:
      "you have heart disease"  → "[medically reviewed]"
      "definitely diagnosed"    → "[medically reviewed]"
    """
    for pattern in _UNSAFE_PATTERNS:
        text = re.sub(pattern, "[medically reviewed]", text, flags=re.IGNORECASE)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Dynamic Prompt Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(
    probability: float,
    decision: str,
    ui_risk_level: str,
    top_features: list,
) -> str:
    """
    Build a dynamic, structured English LLM prompt.

    Parameters
    ----------
    probability   : float  — 0–100 probability score
    decision      : str    — "low" | "high"  (system logic)
    ui_risk_level : str    — "Low Risk" | "Moderate Risk" | "High Risk"
    top_features  : list   — [(feature_name, shap_value), ...] top 3

    Returns
    -------
    str : Complete formatted prompt string.
    """
    urgency_instruction = {
        "high": (
            "Use an urgent and cautionary tone. Strongly emphasize the need for "
            "immediate medical consultation. Prioritize actionable, time-sensitive recommendations."
        ),
        "low": (
            "Use a reassuring and positive tone. Focus on preventive recommendations "
            "and healthy lifestyle reinforcement. Avoid causing unnecessary alarm."
        ),
    }.get(decision, "Use a neutral, objective tone.")

    features_str = "\n".join(
        f"  - {name}: impact score {val:+.3f} "
        f"({'increases risk' if val > 0 else 'decreases risk'})"
        for name, val in top_features
    )

    return f"""
You are an expert AI medical assistant specializing in cardiovascular disease.
Your task is to write a concise, evidence-based medical report summary in English.

Patient Analysis Data:
  - Heart disease probability: {probability:.1f}%
  - Risk classification: {ui_risk_level}
  - Top influencing features (SHAP values):
{features_str}

Tone & Style Instructions:
  {urgency_instruction}

Mandatory Writing Rules:
  1. Always use probabilistic language: "may suggest", "could indicate", "is recommended"
  2. Never state definitively that the patient "has" or "is diagnosed with" heart disease
  3. Do not include statistics or numbers not provided in the input above
  4. Keep the explanation concise: 2-3 sentences only
  5. Recommendations must be practical and specific: 3-5 bullet points
  6. Write in clear, patient-friendly English
"""


# ─────────────────────────────────────────────────────────────────────────────
#  LLM Consultant
# ─────────────────────────────────────────────────────────────────────────────

class HeartDiseaseConsultant:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.0,
            max_tokens=800,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
        )
        self.output_parser = JsonOutputParser(pydantic_object=MedicalReport)

        system_msg = (
            "You are an expert cardiovascular disease consultant. "
            "Write the medical report in clear English as a JSON object per the instructions.\n"
            "{format_instructions}"
        )
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human",  "{user_prompt}"),
        ])
        self._chain = self._prompt | self.llm | self.output_parser

    def generate_report(
        self,
        probability:   float,
        decision:      str,
        ui_risk_level: str,
        top_features:  list,
    ) -> Dict:
        """
        Generate an English medical explanation and recommendations.

        Parameters
        ----------
        probability   : float — 0–100 score
        decision      : str   — "low" | "high"
        ui_risk_level : str   — display label
        top_features  : list  — [(feature, shap_value), ...] top 3

        Returns
        -------
        dict with keys:
            "explanation"     : str
            "recommendations" : list[str]
        """
        try:
            prompt_text = build_prompt(probability, decision, ui_risk_level, top_features)

            raw = self._chain.invoke({
                "user_prompt":          prompt_text,
                "format_instructions":  self.output_parser.get_format_instructions(),
            })

            # Apply medical safety sanitizer
            explanation     = sanitize_llm_output(str(raw.get("explanation", "")))
            recommendations = [
                sanitize_llm_output(r) for r in raw.get("recommendations", [])
            ]

            return {
                "explanation":     explanation,
                "recommendations": recommendations,
            }

        except Exception as e:
            return {
                "explanation":     f"Could not generate explanation: {str(e)}",
                "recommendations": ["Please consult your physician for personalized recommendations."],
            }