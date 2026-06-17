import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure the 'app' directory is in sys.path to resolve 'api', 'services', 'db', etc.
_app_dir = str(Path(__file__).resolve().parent)
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

# Must run before any project import that reads os.environ (e.g. db.database -> core.config).
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path, encoding="utf-8-sig")

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from db.database import get_db

from core.logger import log
from api.router import api_router

if not (os.getenv("INTERNAL_API_KEY") or "").strip():
    print(
        f"WARNING: INTERNAL_API_KEY is missing or empty. "
        f"Set it in {_env_path} (same value as Node INTERNAL_API_KEY), then restart uvicorn."
    )

app = FastAPI(title="Heart Disease Prediction API (Internal)")

# Trusted hosts: default * (all hosts); override with AI_ALLOWED_HOSTS=host1,host2
_allowed = os.getenv("AI_ALLOWED_HOSTS", "*").split(",")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=[h.strip() for h in _allowed if h.strip()])


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    detail = exc.detail
    message = detail if isinstance(detail, str) else str(detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "message": message, "errors": []},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    log.error(f"Unhandled Exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": "Internal Server Error", "errors": []},
    )

@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    statuses = {
        "status": "ok",
        "database": "unknown",
        "ml_local": "unknown",
        "llm": "unknown",
        "ecg": "unknown"
    }
    try:
        db.execute(text("SELECT 1"))
        statuses["database"] = "ok"
    except Exception as e:
        statuses["database"] = "error"
        statuses["status"] = "degraded"
    
    try:
        from services.local_ml_service import local_ml_service
        if local_ml_service.model is not None:
            statuses["ml_local"] = "ok"
        else:
            statuses["ml_local"] = "error"
            statuses["status"] = "degraded"
    except Exception:
        statuses["ml_local"] = "error"
        statuses["status"] = "degraded"

    try:
        if os.getenv("GROQ_API_KEY"):
            statuses["llm"] = "ok"
        else:
            statuses["llm"] = "missing_key"
            statuses["status"] = "degraded"
    except Exception:
        statuses["llm"] = "error"

    try:
        from services.ecg_service import get_ecg_predictor
        predictor = get_ecg_predictor()
        if predictor.model is not None:
            statuses["ecg"] = "ok"
        else:
            statuses["ecg"] = "error"
            statuses["status"] = "degraded"
    except Exception:
        statuses["ecg"] = "error"
        statuses["status"] = "degraded"

    status_code = 200 if statuses["status"] == "ok" else 503
    return JSONResponse(status_code=status_code, content=statuses)


app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
