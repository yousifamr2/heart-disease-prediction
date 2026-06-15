import pandas as pd
import requests
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from services.risk_classifier import assess_risk, RiskAssessment

API_URL = "https://omarbm52-artemis-heart-api.hf.space/predict"

class MLService:
    def __init__(self):
        self.required_cols = [
            "age", "sex", "chest pain type", "resting bp s", "cholesterol",
            "fasting blood sugar", "resting ecg", "max heart rate",
            "exercise angina", "oldpeak", "ST slope"
        ]

    def _normalize_shap_dict(self, raw):
        """Coerce SHAP payload to float values so charts / lru_cache stay hashable and matplotlib-safe."""
        default = {col: 0.1 for col in self.required_cols}
        if not isinstance(raw, dict):
            return default.copy()
        out = {}
        for k, v in raw.items():
            key = str(k)
            try:
                if isinstance(v, (list, tuple)):
                    v = float(v[0]) if len(v) else 0.0
                else:
                    v = float(v)
            except (TypeError, ValueError, IndexError):
                v = 0.0
            out[key] = v
        for col in self.required_cols:
            if col not in out:
                out[col] = 0.1
        return out

    def _prepare_payload(self, data: list):
        return {
            "age": float(data[0]),
            "sex": int(data[1]),
            "chest pain type": int(data[2]),
            "resting bp s": float(data[3]),
            "cholesterol": float(data[4]),
            "fasting blood sugar": int(data[5]),
            "resting ecg": int(data[6]),
            "max heart rate": float(data[7]),
            "exercise angina": int(data[8]),
            "oldpeak": float(data[9]),
            "ST slope": int(data[10])
        }

    def _call_api(self, data: list) -> dict:
        """Single API call — returns raw JSON response dict."""
        payload = self._prepare_payload(data)
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json()

    # ── Binary prediction (0 or 1) ────────────────────────────────────
    def predict_single(self, data: list) -> int:
        try:
            result = self._call_api(data)
            return int(result.get("prediction", 0))
        except Exception as e:
            print(f"HuggingFace unavailable — switched to local model (predict_single): {e}")
            from services.local_ml_service import local_ml_service
            result = local_ml_service.predict_and_explain(data)
            return int(result.get("prediction", 0))

    def predict_dataframe(self, df: pd.DataFrame):
        return [self.predict_single(row.tolist()) for _, row in df.iterrows()]

    # ── Risk + SHAP (legacy — kept for compatibility) ─────────────────
    def get_risk_and_shap(self, data: list):
        """Returns (risk_score_pct, shap_data_dict)."""
        shap_data = {col: 0.1 for col in self.required_cols}
        risk_score = 0.0
        try:
            try:
                result = self._call_api(data)
            except Exception as e:
                print(f"HuggingFace unavailable — switched to local model (get_risk_and_shap): {e}")
                from services.local_ml_service import local_ml_service
                result = local_ml_service.predict_and_explain(data)
            risk_score = float(result.get("probability", 0.0))
            if "shap_values" in result:
                shap_data = result["shap_values"]
        except Exception as e:
            print("API Error in get_risk_and_shap:", e)
        return risk_score, shap_data

    # ── Full hybrid assessment (NEW — preferred) ──────────────────────
    def assess_full_prediction(self, data: list, probability: float = None):
        """
        Calculates or retrieves prediction results.
        If probability is provided (as a percentage 0-100), it uses it directly.
        Otherwise, it calls the model API.
        """
        shap_data = {col: 0.1 for col in self.required_cols}
        try:
            if probability is None:
                # Need fresh prediction from API
                try:
                    result = self._call_api(data)
                except Exception as e:
                    print(f"HuggingFace unavailable — switched to local model (assess_full_prediction): {e}")
                    from services.local_ml_service import local_ml_service
                    result = local_ml_service.predict_and_explain(data)
                    
                probability_pct = float(result.get("probability", 0.0))
                shap_data = self._normalize_shap_dict(
                    result.get("shap_values", shap_data)
                )
            else:
                # Use existing probability (skip API call)
                probability_pct = probability

            assessment = assess_risk(probability_pct / 100.0)
        except Exception as e:
            print("Error in assess_full_prediction:", e)
            assessment = assess_risk(0.0)   # safe fallback
        shap_data = self._normalize_shap_dict(shap_data)
        return assessment, shap_data

    # ── SHAP image generator ──────────────────────────────────────────
    def generate_shap_image(self, shap_data: dict) -> bytes:
        features   = list(shap_data.keys())
        importance = [abs(v) for v in shap_data.values()]

        shap_df = pd.DataFrame({
            "feature":    features,
            "importance": importance
        }).sort_values(by="importance", ascending=False)

        plt.figure(figsize=(8, 4))
        plt.barh(shap_df["feature"], shap_df["importance"])
        plt.gca().invert_yaxis()
        plt.title("Feature Importance (SHAP)")
        plt.xlabel("Feature Importance")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return buf.read()


ml_service = MLService()
