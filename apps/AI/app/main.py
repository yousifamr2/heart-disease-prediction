import os
<<<<<<< HEAD
from pathlib import Path

from dotenv import load_dotenv

# Must run before any project import that reads os.environ (e.g. db.database -> core.config).
_env_path = Path(__file__).resolve().parent.parent / ".env"
=======
import sys
from pathlib import Path

# Add the 'app' directory and its parent 'AI' directory to sys.path to resolve all import styles
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.append(str(_current_dir))
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.append(str(_parent_dir))

from dotenv import load_dotenv

# Must run before any project import that reads os.environ (e.g. db.database -> core.config).
_env_path = _current_dir.parent / ".env"
>>>>>>> main
load_dotenv(_env_path, encoding="utf-8-sig")

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware

<<<<<<< HEAD
from api.router import api_router
=======
from app.api.router import api_router
>>>>>>> main

if not (os.getenv("INTERNAL_API_KEY") or "").strip():
    print(
        f"WARNING: INTERNAL_API_KEY is missing or empty. "
        f"Set it in {_env_path} (same value as Node INTERNAL_API_KEY), then restart uvicorn."
    )

app = FastAPI(title="Heart Disease Prediction API (Internal)")

# Trusted hosts: default localhost; override with AI_ALLOWED_HOSTS=host1,host2
_allowed = os.getenv("AI_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver").split(",")
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
    print(f"Unhandled Exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": "Internal Server Error", "errors": []},
    )


app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
