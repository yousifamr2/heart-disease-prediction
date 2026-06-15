import pytest
from unittest.mock import patch, MagicMock
import numpy as np

@patch("app.services.ecg_service.xresnet1d101")
@patch("app.services.ecg_service.pickle.load")
@patch("app.services.ecg_service.torch.load")
@patch("app.services.ecg_service._first_pth_under_weights")
def test_ecg_predictor(mock_pth, mock_tload, mock_pload, mock_model_init):
    mock_pth.return_value = MagicMock()
    mock_tload.return_value = {"model": {}}
    
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1000, 12))
    
    mock_mlb = MagicMock()
    mock_mlb.classes_ = ["NORM", "MI", "ST"]
    
    mock_pload.side_effect = [mock_scaler, mock_mlb]
    
    with patch("pathlib.Path.is_file", return_value=True):
        with patch("builtins.open", MagicMock()):
            from app.services.ecg_service import ECGPredictor
            import torch
            
            predictor = ECGPredictor()
            predictor.model = MagicMock()
            predictor.model.return_value = torch.tensor([[0.9, -0.5, 0.1]])
            
            res = predictor.predict(np.zeros((1000, 12)))
            
            assert isinstance(res, list)
            assert len(res) > 0
            assert "label" in res[0]
            assert "probability" in res[0]
