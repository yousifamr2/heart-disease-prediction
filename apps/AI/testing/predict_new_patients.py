"""
predict_new_patients.py
-----------------------
Loads the pre-trained best_model.pkl and predicts heart disease
for 10 unseen patients (5 with heart disease + 5 without).

The model never saw these instances during training.
"""

import os
import joblib
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models",  "best_model.pkl")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

CSV_NO_DISEASE   = os.path.join(DATASET_DIR, "patients_no_heart_disease.csv")
CSV_WITH_DISEASE = os.path.join(DATASET_DIR, "patients_with_heart_disease.csv")

# Features used during training (target column is excluded)
FEATURES = [
    "age", "sex", "chest pain type", "resting bp s", "cholesterol",
    "fasting blood sugar", "resting ecg", "max heart rate",
    "exercise angina", "oldpeak", "ST slope"
]

# ── Load model ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Heart Disease Prediction -- Unseen Patients")
print("=" * 60)

model = joblib.load(MODEL_PATH)
print("\n[OK] Model loaded:", type(model).__name__, "\n")

# ── Load the 10 patients ───────────────────────────────────────────────────────
df_no_disease   = pd.read_csv(CSV_NO_DISEASE)
df_with_disease = pd.read_csv(CSV_WITH_DISEASE)

# Store real labels before dropping them
true_labels_no   = df_no_disease["target"].values
true_labels_with = df_with_disease["target"].values

# Drop target column -- model predicts WITHOUT seeing it
X_no   = df_no_disease[FEATURES]
X_with = df_with_disease[FEATURES]

# ── Predict ────────────────────────────────────────────────────────────────────
preds_no   = model.predict(X_no)
preds_with = model.predict(X_with)

# ── Display results ────────────────────────────────────────────────────────────
LABEL_MAP = {0: "No Heart Disease", 1: "Heart Disease"}

def print_results(predictions, true_labels, group_title):
    print("\n" + "-" * 60)
    print("  " + group_title)
    print("-" * 60)
    for i, (pred, true) in enumerate(zip(predictions, true_labels), start=1):
        status   = "CORRECT" if pred == true else "WRONG"
        pred_lbl = LABEL_MAP[pred]
        true_lbl = LABEL_MAP[true]
        print(f"  Patient {i:>2} | Predicted: {pred_lbl:<18} | Actual: {true_lbl:<18} | {status}")

print_results(preds_no,   true_labels_no,   "Group A -- Patients WITHOUT Heart Disease (5 patients)")
print_results(preds_with, true_labels_with, "Group B -- Patients WITH Heart Disease    (5 patients)")

# ── Summary ────────────────────────────────────────────────────────────────────
all_preds  = list(preds_no)        + list(preds_with)
all_labels = list(true_labels_no)  + list(true_labels_with)
correct    = sum(p == t for p, t in zip(all_preds, all_labels))
total      = len(all_preds)

print("\n" + "=" * 60)
print(f"  Overall Accuracy on 10 Unseen Patients: {correct}/{total} ({correct/total*100:.0f}%)")
print("=" * 60 + "\n")
