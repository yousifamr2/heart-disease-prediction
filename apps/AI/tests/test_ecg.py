import pytest
import io
import numpy as np

def test_ecg_pipeline_invalid_extension(client, mock_headers):
    dat_content = b"fake data"
    hea_content = b"fake hea"
    
    files = {
        "dat_file": ("test.txt", io.BytesIO(dat_content), "text/plain"),
        "hea_file": ("test.txt", io.BytesIO(hea_content), "text/plain")
    }
    data = {"ecg_test_id": "test-id"}
    
    response = client.post("/internal/ecg/pipeline", data=data, files=files, headers=mock_headers)
    assert response.status_code == 422
    assert "Expected a .dat file" in response.json()["message"]

# Due to WFDB complex parsing, we test validation layers in isolated functions if possible,
# or mock WFDB rdrecord in integration tests.
