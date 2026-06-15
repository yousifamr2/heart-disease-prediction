from __future__ import annotations

import io
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import wfdb
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.security import verify_internal_api_key
from data.ecg_diagnosis_kb import build_kb_context_for_top5
from services.chart_service import generate_ecg_top5_chart_png_bytes
from services.ecg_service import get_ecg_predictor
from services.llm_service import ECG_PROMPT_VERSION

router = APIRouter(
    prefix="/internal/ecg",
    tags=["Internal ECG"],
    dependencies=[Depends(verify_internal_api_key)],
)

_ecg_consultant_singleton: Any = None


def _ecg_consultant():
    global _ecg_consultant_singleton
    if _ecg_consultant_singleton is None:
        from services.llm_service import EcgConsultant

        _ecg_consultant_singleton = EcgConsultant()
    return _ecg_consultant_singleton


def _require_wfdb_extensions(name: str, filename: str | None) -> None:
    ext = (Path(filename or "").suffix or "").lower()
    if ext != f".{name}":
        raise HTTPException(status_code=422, detail=f"Expected a .{name} file upload.")


class EcgChartBody(BaseModel):
    top_5: list[dict[str, Any]] = Field(..., description="Top ECG findings with probability %")
    compact: bool = False


class EcgReportPatient(BaseModel):
    name: str = "Patient"
    national_id: str = ""
    email: str = ""


class EcgReportLab(BaseModel):
    name: str = "N/A"
    address: str = "N/A"
    lab_code: str = ""


class EcgReportBody(BaseModel):
    ecg_test: dict[str, Any]
    patient: EcgReportPatient
    lab: EcgReportLab
    top_5: list[dict[str, Any]]
    llm_ecg_json: dict[str, Any] | None = None
    primary_diagnosis: str | None = None
    primary_probability: float | None = None


def _confidence_label(pct: float | None) -> str:
    if pct is None:
        return "Not reported"
    if pct >= 85:
        return "High model confidence (still not a clinical diagnosis)"
    if pct >= 60:
        return "Moderate model confidence"
    if pct >= 35:
        return "Limited model confidence — correlation recommended"
    return "Low model confidence — interpret cautiously"


@router.post("/chart")
def ecg_chart_png(body: EcgChartBody):
    if not body.top_5:
        raise HTTPException(status_code=422, detail="top_5 is required")
    png = generate_ecg_top5_chart_png_bytes(body.top_5, compact=body.compact)
    return StreamingResponse(io.BytesIO(png), media_type="image/png")


@router.post("/report")
def ecg_report_pdf(body: EcgReportBody):
    from services.pdf_service import generate_ecg_medical_report_pdf

    buf = generate_ecg_medical_report_pdf(
        patient=body.patient.model_dump(),
        lab=body.lab.model_dump(),
        ecg_test=body.ecg_test,
        top_5=body.top_5,
        llm_ecg_json=body.llm_ecg_json or {},
        primary_diagnosis=body.primary_diagnosis,
        primary_probability=body.primary_probability,
        confidence_label=_confidence_label(body.primary_probability),
    )
    return StreamingResponse(buf, media_type="application/pdf")


@router.post("/pipeline")
def ecg_pipeline(
    ecg_test_id: str = Form(..., description="EcgTest UUID (gateway / Neon)"),
    dat_file: UploadFile = File(...),
    hea_file: UploadFile = File(...),
):
    """
    WFDB .dat + .hea → model top-5 + LLM interpretation. Does not write to SQLAlchemy DB (gateway persists).
    """
    _require_wfdb_extensions("dat", dat_file.filename)
    _require_wfdb_extensions("hea", hea_file.filename)

    tmp_dir = Path(tempfile.mkdtemp(prefix="ecg_wfdb_"))
    try:
        dat_bytes = dat_file.file.read()
        hea_bytes = hea_file.file.read()

        hea_text = hea_bytes.decode("utf-8", errors="replace").lstrip("\ufeff")
        first_line = (hea_text.splitlines() or [""])[0].strip()
        if not first_line:
            raise HTTPException(status_code=422, detail="Empty or invalid .hea file.")
        record_name = first_line.split()[0]
        if not record_name or any(c in record_name for c in ("/", "\\", "..")):
            raise HTTPException(status_code=422, detail="Invalid WFDB record name in .hea header.")

        dat_path = tmp_dir / f"{record_name}.dat"
        hea_path = tmp_dir / f"{record_name}.hea"
        hea_path.write_bytes(hea_bytes)
        dat_path.write_bytes(dat_bytes)

        record_base = (tmp_dir / record_name).resolve()
        try:
            record = wfdb.rdrecord(str(record_base))
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not read WFDB record: {exc}") from exc

        sig = getattr(record, "p_signal", None)
        if sig is None:
            raise HTTPException(status_code=422, detail="Record has no physical signal (p_signal).")

        sig = np.asarray(sig, dtype=np.float32)
        if sig.ndim != 2:
            raise HTTPException(status_code=422, detail=f"Unexpected signal rank: {sig.ndim}")

        n_sig = sig.shape[1]
        if n_sig < 12:
            raise HTTPException(
                status_code=422,
                detail=f"Record has {n_sig} signal channel(s); at least 12 are required.",
            )
            
        # 1. Validate Lead Order
        expected_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        sig_names = getattr(record, "sig_name", None)
        if not sig_names or len(sig_names) < 12:
            raise HTTPException(status_code=422, detail="Missing or incomplete lead names in WFDB record.")
        actual_leads = [str(x).strip() for x in sig_names[:12]]
        if actual_leads != expected_leads:
            raise HTTPException(status_code=422, detail=f"Invalid lead order. Expected: {expected_leads}, Got: {actual_leads}")

        if n_sig > 12:
            sig = sig[:, :12]
            
        # 2. Signal Quality & Length Validation
        MAX_SIGNAL_LENGTH = 10000
        if sig.shape[0] > MAX_SIGNAL_LENGTH:
            raise HTTPException(status_code=422, detail=f"Signal length {sig.shape[0]} exceeds maximum allowed ({MAX_SIGNAL_LENGTH}).")
            
        if np.isnan(sig).any() or np.isinf(sig).any():
            raise HTTPException(status_code=422, detail="Signal contains NaN or infinite values.")
            
        # 3. Resampling
        import scipy.signal
        fs = getattr(record, "fs", None)
        TARGET_FS = 100
        if fs is not None and fs != TARGET_FS:
            num_samples = int(sig.shape[0] * TARGET_FS / fs)
            sig = scipy.signal.resample(sig, num_samples, axis=0)
            sig = np.asarray(sig, dtype=np.float32)

        predictor = get_ecg_predictor()
        top_5 = predictor.predict(sig)

        if not top_5:
            raise HTTPException(
                status_code=422,
                detail="Model returned no non-zero findings; cannot derive a primary diagnosis.",
            )

        codes = [str(x.get("code") or "") for x in top_5[:5]]
        kb_context = build_kb_context_for_top5(top_5)
        llm_raw = _ecg_consultant().generate_ecg_report(top_5, kb_context)

        llm_ecg_json = {
            "type": "ecg_llm_multilabel",
            "prompt_version": ECG_PROMPT_VERSION,
            "ecg_test_id": ecg_test_id,
            "top_5_codes": codes,
            **llm_raw,
        }

        return {
            "success": True,
            "n_samples": int(sig.shape[0]),
            "top_5": top_5,
            "model_name": "xresnet1d101",
            "model_version": "ptbxl_style",
            "llm_model": "llama-3.3-70b-versatile",
            "llm_prompt_version": ECG_PROMPT_VERSION,
            "llm_ecg_json": llm_ecg_json,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
