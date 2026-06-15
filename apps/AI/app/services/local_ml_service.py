import joblib
import pandas as pd
import shap
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "best.pkl"

class LocalMLService:
    def __init__(self):
        self.model = None
        self.explainer = None
        self.feature_names = [
            "age", "sex", "chest pain type", "resting bp s", "cholesterol",
            "fasting blood sugar", "resting ecg", "max heart rate",
            "exercise angina", "oldpeak", "ST slope"
        ]
        self._load_model()

    def _load_model(self):
        try:
            self.model = joblib.load(MODEL_PATH)
            # Create a SHAP explainer for the tree model (CatBoost)
            self.explainer = shap.TreeExplainer(self.model)
            print(f"Successfully loaded local model from {MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load local model from {MODEL_PATH}: {e}")

    def predict_and_explain(self, data: list) -> dict:
        if self.model is None:
            raise RuntimeError("Local model is not loaded.")
        
        # Format data as DataFrame to match expected feature names
        df = pd.DataFrame([data], columns=self.feature_names)
        
        # Predict probability
        proba = self.model.predict_proba(df)
        prob = proba[0][1] if len(proba[0]) > 1 else proba[0][0]
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # Convert to dictionary
        shap_dict = {}
        for i, feature in enumerate(self.feature_names):
            shap_dict[feature] = float(shap_values[0][i])
            
        return {
            "prediction": 1 if prob >= 0.41 else 0,
            "probability": float(prob) * 100.0,
            "shap_values": shap_dict
        }

local_ml_service = LocalMLService()
