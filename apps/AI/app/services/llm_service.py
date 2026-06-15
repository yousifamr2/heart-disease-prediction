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
    risk_level: str = Field(description="The exact risk level matching the input UI risk level (e.g., Low Risk, Moderate Risk, High Risk)")
    key_factors: List[str] = Field(description="The exact list of SHAP feature names that are provided in the input, without changes")
    explanation: str = Field(description="2-3 simple sentences in plain English explaining the risk level and explicitly mentioning every feature in the key factors list")
    recommendations: List[str] = Field(description="Exactly 5 specific, actionable wellness recommendations in English, each linked to at least one feature")
    medical_disclaimer: str = Field(description="Standard disclaimer: 'This result is not a diagnosis. Please consult a doctor for medical advice.'")


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
You are an expert AI medical assistant specializing in heart health.
Your task is to write a concise, evidence-based medical report summary in English.

Patient Analysis Data:
  - Heart disease probability: {probability:.1f}%
  - Risk classification: {ui_risk_level}
  - Top influencing features (SHAP values):
{features_str}

Tone & Style Instructions:
  {urgency_instruction}

Mandatory Writing Rules:
  1. Always use probabilistic language: "may suggest", "could indicate", "is recommended".
  2. Never state definitively that the patient "has" or "is diagnosed with" heart disease, and never claim arterial blockage.
  3. Do not include statistics or numbers not provided in the input above.
  4. Keep the explanation concise: exactly 2-3 sentences.
  5. Recommendations must be practical and specific: EXACTLY 5 bullet points.
  6. Write in clear, simple, patient-friendly English (6th-to-8th grade reading level, Flesch Reading Ease score of 65+).
  7. Use short sentences (under 15 words) and simple vocabulary: use 'heart' (not 'cardiovascular'), 'doctor' (not 'cardiologist'), 'high blood pressure' (not 'hypertension').
  8. Safety Rule: Never prescribe medication or specify drug names (do NOT mention 'Lipitor' or any specific drug brand, even to refuse it. Refer to them generally as 'medication' or 'prescriptions').
  9. Grounding Rule: In your explanation, you must explicitly mention the risk classification level (e.g. 'Low Risk', 'Moderate Risk', or 'High Risk') and name every feature in the input SHAP list (e.g. 'age', 'resting ecg', 'cholesterol', 'ST slope', 'oldpeak', 'exercise angina', 'max heart rate', 'resting bp s', or 'chest pain type') exactly as they are written.
  10. Recommendation Anchors: If the decision is 'high', your recommendations must contain at least one of these anchor words: 'doctor', 'physician', or 'medical evaluation'. If the decision is 'low', your recommendations must contain at least one of these anchor words: 'lifestyle', 'preventive', 'exercise', or 'healthy diet'.
  11. Evidence-Based link: Every recommendation should be linked to at least one reported feature (e.g. 'Regular exercise may help manage resting bp s').
  12. Medical Disclaimer: Always include the exact text: 'This result is not a diagnosis. Please consult a doctor for medical advice.'
"""


# ─────────────────────────────────────────────────────────────────────────────
#  LLM Consultant
# ─────────────────────────────────────────────────────────────────────────────

class HeartDiseaseConsultant:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.2,
            max_tokens=800,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            model_kwargs={"top_p": 0.9}
        )
        self.output_parser = JsonOutputParser(pydantic_object=MedicalReport)

        system_msg = (
            "You are an expert heart health consultant. Write the medical report in clear, simple, patient-friendly English "
            "as a JSON object per the instructions.\n"
            "Strict Safety Rules:\n"
            "- Do NOT prescribe medications or specify drug names (do NOT mention 'Lipitor' or any other specific drug name, even to refuse it. Refer to them generally as 'medication' or 'prescriptions').\n"
            "- Do NOT confirm specific blockages or diagnoses; always state that this is computer-aided risk assessment only.\n"
            "Strict Grounding Rules:\n"
            "- Only refer to the patient data provided in the prompt. Do not introduce any other clinical metrics or patient details.\n"
            "- Always explicitly state the risk classification and feature names (like 'age', 'resting ecg', 'cholesterol', 'st slope', 'oldpeak', 'exercise angina', 'max heart rate', 'resting bp s', 'chest pain type') in your explanation.\n"
            "Strict Readability Rules:\n"
            "- Write in a plain-English, conversational, simple tone. Use short sentences (under 15 words) and simple words.\n"
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
                "risk_level":      str(raw.get("risk_level", ui_risk_level)),
                "key_factors":     raw.get("key_factors", [f[0] for f in top_features]),
                "explanation":     explanation,
                "recommendations": recommendations,
                "medical_disclaimer": str(raw.get("medical_disclaimer", "This result is not a diagnosis. Please consult a doctor for medical advice.")),
            }

        except Exception as e:
            return {
                "explanation":     f"Could not generate explanation: {str(e)}",
                "recommendations": ["Please consult your physician for personalized recommendations."],
            }


ECG_PROMPT_VERSION = "ecg_v1"


class EcgMultiLabelReport(BaseModel):
    interpretation: str = Field(
        description="3-5 sentences: what the automated ECG findings may suggest, in probabilistic language"
    )
    urgency: str = Field(description="One short paragraph on how soon to seek medical advice (non-alarmist)")
    follow_up: str = Field(description="Concrete follow-up steps (primary care, cardiology, ED if red flags)")
    warning_signs: List[str] = Field(
        description="3-6 bullet-style short strings of symptoms that should prompt urgent care"
    )
    recommendations: List[str] = Field(
        description="5-7 lifestyle and medical adherence recommendations; no definitive diagnosis language"
    )


def build_ecg_prompt(top_5_lines: str, kb_context: str, primary_line: str) -> str:
    return f"""
You are an expert cardiologist assistant helping explain **automated multi-label ECG classifier output**.
This is computer-aided analysis only — **not** a definitive diagnosis.

Primary finding (highest model output):
{primary_line}

Top-5 model outputs (label, probability %):
{top_5_lines}

Reference context for the listed SCP-style codes (educational, for you — do not quote verbatim as patient diagnosis):
{kb_context}

Mandatory rules:
1. Use probabilistic phrasing: "may suggest", "could be consistent with", "warrants correlation with", "is recommended".
2. Never state the patient definitively "has" a disease based on this ECG alone.
3. Mention that automated classifiers can be wrong and that electrode placement, artifact, and patient factors matter.
4. Address urgency responsibly: chest pain, syncope, severe dyspnea → emergency care.
5. Do not invent numeric probabilities beyond those given above.
6. Keep tone calm, precise, and patient-friendly.

Return ONLY valid JSON matching the schema instructions.
"""


class EcgConsultant:
    """LLM layer for ECG multi-label interpretation."""

    def __init__(self) -> None:
        self.llm = ChatGroq(
            temperature=0.0,
            max_tokens=1200,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
        )
        self.output_parser = JsonOutputParser(pydantic_object=EcgMultiLabelReport)
        system_msg = (
            "You explain automated ECG classifier results responsibly. Output JSON only.\n"
            "{format_instructions}"
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),
                ("human", "{user_prompt}"),
            ]
        )
        self._chain = self._prompt | self.llm | self.output_parser

    def generate_ecg_report(self, top_5: list, kb_context: str) -> Dict[str, object]:
        try:
            primary = top_5[0] if top_5 else {}
            primary_line = (
                f"- {primary.get('label', primary.get('code', ''))}: {primary.get('probability', '')}%"
                if primary
                else "(none)"
            )
            top_5_lines = "\n".join(
                f"- {x.get('label', x.get('code', '?'))}: {x.get('probability', '')}% (code {x.get('code', '')})"
                for x in top_5[:5]
            )
            prompt_text = build_ecg_prompt(top_5_lines, kb_context, primary_line)
            raw = self._chain.invoke(
                {
                    "user_prompt": prompt_text,
                    "format_instructions": self.output_parser.get_format_instructions(),
                }
            )
            interpretation = sanitize_llm_output(str(raw.get("interpretation", "")))
            urgency = sanitize_llm_output(str(raw.get("urgency", "")))
            follow_up = sanitize_llm_output(str(raw.get("follow_up", "")))
            warning_signs = [sanitize_llm_output(str(w)) for w in raw.get("warning_signs", []) if str(w).strip()]
            recommendations = [sanitize_llm_output(str(r)) for r in raw.get("recommendations", []) if str(r).strip()]
            return {
                "interpretation": interpretation,
                "urgency": urgency,
                "follow_up": follow_up,
                "warning_signs": warning_signs,
                "recommendations": recommendations,
            }
        except Exception as e:
            return {
                "interpretation": (
                    "We could not generate an extended narrative interpretation. "
                    f"Technical detail: {str(e)[:200]}"
                ),
                "urgency": "If you have chest pain, fainting, or severe shortness of breath, seek emergency care.",
                "follow_up": "Share this automated ECG summary with your physician for clinical correlation.",
                "warning_signs": [
                    "Pressure-like chest pain",
                    "Fainting or near-fainting",
                    "Severe shortness of breath",
                    "Rapid irregular heartbeat with symptoms",
                ],
                "recommendations": [
                    "Follow heart-healthy diet guidance from your clinician.",
                    "Stay physically active as approved by your physician.",
                    "Avoid tobacco and limit alcohol per medical advice.",
                    "Ensure blood pressure and glucose are managed if applicable.",
                    "Discuss whether ambulatory monitoring is appropriate.",
                ],
            }