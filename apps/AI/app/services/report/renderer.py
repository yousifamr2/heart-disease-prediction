"""
report/renderer.py
──────────────────────────────────────────────────────────────────────────
HTML Rendering Layer — injects English report context into the Jinja2 template.

Single responsibility: context dict → HTML string (English only).
No Arabic processing, no PDF logic, no LLM calls.
"""

from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=select_autoescape(["html"]),
)


def render_report(context: dict) -> str:
    """
    Render the English medical report HTML template with the given context.

    Parameters
    ----------
    context : dict
        patient_info     : dict  (label → value)
        probability      : float (0–100)
        decision         : str   ("low" | "high")
        ui_risk_level    : str   ("Low Risk" | "Moderate Risk" | "High Risk")
        ui_risk_color    : str   (hex color string)
        feat_chart_b64   : str   (base64 data-URI)
        shap_chart_b64   : str   (base64 data-URI)
        explanation      : str   (LLM-generated, English)
        recommendations  : list[str] (LLM-generated, English)
        generated_at     : str   (formatted datetime)

    Returns
    -------
    str : Fully rendered HTML string, ready for xhtml2pdf conversion.
    """
    template = _env.get_template("medical_report.html")
    return template.render(**context)
