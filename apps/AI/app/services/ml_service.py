import joblib
import shap
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import io

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "best_model.pkl"

class MLService:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.explainer = shap.Explainer(self.model)
        self.required_cols = [
            "age", "sex", "chest pain type", "resting bp s", "cholesterol",
            "fasting blood sugar", "resting ecg", "max heart rate",
            "exercise angina", "oldpeak", "ST slope"
        ]

    def predict_single(self, data: list) -> int:
        prediction = self.model.predict([data])[0]
        return int(prediction)

    def predict_dataframe(self, df: pd.DataFrame):
        return self.model.predict(df)

    def generate_shap_image(self, data: list) -> bytes:
        df_data = pd.DataFrame([data], columns=self.required_cols)
        shap_values = self.explainer(df_data)

        values = shap_values.values[0, :, 1]
        features = df_data.columns

        shap_df = pd.DataFrame({
            "feature": features,
            "importance": abs(values)
        }).sort_values(by="importance", ascending=False)

        plt.figure(figsize=(8,4))
        plt.barh(shap_df["feature"], shap_df["importance"])
        plt.gca().invert_yaxis()
        plt.title("Feature Importance (SHAP)")
        plt.xlabel("Importance")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()

        buf.seek(0)
        return buf.read()

ml_service = MLService()
