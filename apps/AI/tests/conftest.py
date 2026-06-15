import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os

# Set environment variables before importing the app
os.environ["INTERNAL_API_KEY"] = "test_key"
os.environ["GROQ_API_KEY"] = "test_groq_key"

from app.main import app

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test_key"}
