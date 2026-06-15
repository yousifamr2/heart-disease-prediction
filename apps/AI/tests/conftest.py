import pytest
from fastapi.testclient import TestClient
<<<<<<< Updated upstream
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
=======
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import sys

# Insert the app directory to sys.path so imports like 'from db.database' work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

os.environ["INTERNAL_API_KEY"] = "test-secret"
os.environ["GROQ_API_KEY"] = "fake-groq-key"

from main import app
from db.database import Base, get_db
from db.models import LabTest, Prediction, User, Lab

# Use SQLite for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="module")
def db():
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="module")
def client(db):
    def override_get_db():
        try:
            yield db
        finally:
            pass
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c

@pytest.fixture
def mock_headers():
    return {"X-INTERNAL-API-KEY": "test-secret"}
>>>>>>> Stashed changes
