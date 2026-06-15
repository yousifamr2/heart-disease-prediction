"""
Internal service authentication.

Only the Node.js API gateway should call FastAPI with X-INTERNAL-API-KEY.
Never expose this key to browsers or mobile clients.
"""

import os
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-INTERNAL-API-KEY", auto_error=False)


def verify_internal_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> str:
    expected = (os.getenv("INTERNAL_API_KEY") or "").strip()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service INTERNAL_API_KEY is not configured",
        )
    if not api_key or not secrets.compare_digest(api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )
    return api_key
