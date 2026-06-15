import pytest

def test_missing_api_key(client):
    response = client.post("/internal/predict", json={"target_id": "test", "user_id": "test"})
    assert response.status_code == 401
    assert response.json() == {"success": False, "message": "Unauthorized", "errors": []}

def test_invalid_api_key(client):
    response = client.post("/internal/predict", json={"target_id": "test", "user_id": "test"}, headers={"X-INTERNAL-API-KEY": "wrong-key"})
    assert response.status_code == 401
    assert response.json() == {"success": False, "message": "Unauthorized", "errors": []}

def test_valid_api_key_no_patient(client, mock_headers):
    response = client.post("/internal/predict", json={"target_id": "non-existent", "user_id": "test"}, headers=mock_headers)
    assert response.status_code == 404
    assert response.json() == {"success": False, "message": "LabTest not found", "errors": []}
