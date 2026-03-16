"""
Streamlit Web App for Heart Disease Prediction
----------------------------------------------
Uses a pre-trained model saved to disk (e.g., models/best_model.pkl).

Features:
- Upload CSV, validate columns, reorder to training schema
- Predict labels and probabilities (if supported)
- Preview input data and results
- Download predictions as CSV
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------
# Utility functions
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_model(model_path: str):
    """
    Load the trained model from disk.
    Handles both raw models and sklearn Pipeline objects safely.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    logger.info(f"Model loaded successfully from {model_path}")
    
    # Check if it's a Pipeline
    if hasattr(model, 'named_steps'):
        logger.info("Detected sklearn Pipeline")
    
    return model


def infer_feature_names(model):
    """
    Infer feature names from the trained model if available.
    Handles both raw models and Pipeline objects.
    Falls back to None when not present.
    """
    # Try direct attribute first
    if hasattr(model, "feature_names_in_"):
        logger.info(f"Found {len(model.feature_names_in_)} features from model.feature_names_in_")
        return model.feature_names_in_
    
    # Try Pipeline's named_steps
    if hasattr(model, 'named_steps'):
        # Check if there's a preprocessor step
        if 'preprocessor' in model.named_steps:
            preprocessor = model.named_steps['preprocessor']
            if hasattr(preprocessor, 'feature_names_in_'):
                logger.info("Found features from Pipeline preprocessor")
                return preprocessor.feature_names_in_
        
        # Check final estimator
        for step_name, step_obj in model.named_steps.items():
            if hasattr(step_obj, 'feature_names_in_'):
                logger.info(f"Found features from Pipeline step: {step_name}")
                return step_obj.feature_names_in_
    
    logger.warning("Could not infer feature names from model")
    return None


def validate_and_prepare(data: pd.DataFrame, required_features: list):
    """
    Validate uploaded data has required features and reorder columns.
    Returns cleaned dataframe and list of missing columns (if any).
    """
    missing = [col for col in required_features if col not in data.columns]
    if missing:
        return None, missing

    # Drop extra columns, keep only required in correct order
    prepared = data[required_features].copy()
    return prepared, []


def load_scaler_if_exists():
    """
    Try to load a saved scaler from common paths.
    Returns scaler object or None if not found.
    """
    possible_scaler_paths = [
        "AI/models/scaler.pkl",
        "models/scaler.pkl",
        "./AI/models/scaler.pkl",
        "./models/scaler.pkl"
    ]
    
    for path in possible_scaler_paths:
        if os.path.exists(path):
            try:
                scaler = joblib.load(path)
                logger.info(f"Loaded scaler from {path}")
                return scaler
            except Exception as e:
                logger.warning(f"Failed to load scaler from {path}: {e}")
    
    return None


def predict(model, X: pd.DataFrame, threshold: float = 0.7):
    """
    Generate predictions and probabilities (if supported).
    Uses custom threshold (default 70%) for classification.
    
    Parameters:
    -----------
    model : trained model
        The trained model to use for predictions
    X : pd.DataFrame
        Input features
    threshold : float
        Probability threshold for Disease classification (default: 0.7)
        If probability of Disease >= threshold, predict Disease (1)
        Otherwise, predict No Disease (0)
    """
    try:
        # Check if model is NOT a Pipeline and try to load scaler
        if not hasattr(model, 'named_steps'):
            scaler = load_scaler_if_exists()
            if scaler is not None:
                try:
                    X_scaled = scaler.transform(X)
                    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
                    logger.info("Applied scaler to input data")
                except Exception as e:
                    logger.warning(f"Failed to apply scaler: {e}. Proceeding without scaling.")
        
        # Get probabilities first
        proba = None
        preds = None
        
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
                
                # Apply custom threshold (70% by default)
                if proba.shape[1] >= 2:
                    disease_proba = proba[:, 1]  # Probability of Disease class
                    preds = (disease_proba >= threshold).astype(int)
                    logger.info(f"Applied threshold {threshold*100:.0f}%: {preds.sum()} predicted as Disease out of {len(preds)}")
                else:
                    # Fallback to default predict if probabilities shape is unexpected
                    preds = model.predict(X)
                    logger.warning("Using default predict() due to unexpected probability shape")
            except Exception as e:
                logger.warning(f"Could not generate probabilities: {e}, using default predict()")
                preds = model.predict(X)
        else:
            # If model doesn't support predict_proba, use default predict
            preds = model.predict(X)
            logger.warning("Model does not support predict_proba, using default threshold (50%)")
        
        return preds, proba
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("Heart Disease Prediction")
st.write(
    "Upload a CSV file with the same feature columns used during training "
    "to predict heart disease (0 = No Disease, 1 = Disease)."
)

# Sidebar
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Prediction threshold slider (default 70%)
st.sidebar.subheader("Prediction Settings")
prediction_threshold = st.sidebar.slider(
    "Probability Threshold for Disease (%)",
    min_value=0,
    max_value=100,
    value=70,
    step=5,
    help="If probability of Disease >= threshold, predict Disease. Otherwise, predict No Disease."
)
prediction_threshold_decimal = prediction_threshold / 100.0

# Try multiple possible model paths
possible_paths = [
    "AI/models/best_model.pkl",
    "models/best_model.pkl",
    "./AI/models/best_model.pkl",
    "./models/best_model.pkl"
]

model_path = None
for path in possible_paths:
    if os.path.exists(path):
        model_path = path
        break

if model_path is None:
    st.sidebar.error("Model file not found. Please ensure best_model.pkl exists in AI/models/ or models/ directory.")
    st.stop()

metric_info = f"Model loaded from: {model_path}"

# Load model once
try:
    model = load_model(model_path)
    st.sidebar.success(f"Model loaded successfully from: {model_path}")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    st.stop()

# Infer expected features from model
expected_features = infer_feature_names(model)
if expected_features is None:
    st.warning(
        "Could not infer feature names from the model. "
        "Ensure the uploaded CSV has the exact training columns in the correct order."
    )


def process_file(file) -> pd.DataFrame:
    """Read the uploaded CSV into a DataFrame."""
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None


if uploaded_file:
    df_raw = process_file(uploaded_file)
    if df_raw is not None:
        # Check if target column is present and remove it with warning
        if 'target' in df_raw.columns:
            st.warning("⚠️ Target column detected in uploaded CSV. Removing it for prediction.")
            df_raw = df_raw.drop(columns=['target'])
            logger.info("Dropped target column from uploaded data")
        
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df_raw.head())

        # Validate columns if we know expected features
        if expected_features is not None:
            X_prepared, missing_cols = validate_and_prepare(df_raw, list(expected_features))
            if missing_cols:
                st.error(
                    f"❌ Missing required columns: {missing_cols}. "
                    "Please upload a CSV with the full feature set."
                )
                logger.error(f"Missing columns: {missing_cols}")
                st.stop()
            
            # Ensure columns are in expected order
            X_prepared = X_prepared[list(expected_features)]
            logger.info("Data validated and prepared successfully")
        else:
            # If we can't infer, use uploaded columns and warn user
            st.warning(
                "⚠️ Could not infer feature names from model. "
                "Using uploaded CSV columns as-is. Ensure they match training order."
            )
            X_prepared = df_raw.copy()
            logger.warning("Proceeding with uploaded columns (no feature validation)")

        # Prediction
        if st.sidebar.button("Predict"):
            try:
                with st.spinner("🔮 Predicting..."):
                    preds, proba = predict(model, X_prepared, threshold=prediction_threshold_decimal)

                # Build results DataFrame
                results = df_raw.copy()
                results["prediction"] = preds
                results["prediction_label"] = np.where(preds == 1, "Disease", "No Disease")

                if proba is not None and proba.shape[1] >= 2:
                    results["prob_no_disease"] = proba[:, 0]
                    results["prob_disease"] = proba[:, 1]

                st.success(f"✅ Predictions generated successfully for {len(results)} samples!")
                
                # Display threshold info
                st.info(f"📊 Using threshold: {prediction_threshold}% (Probability >= {prediction_threshold}% → Disease)")
                
                st.subheader("Prediction Results")
                st.dataframe(results.head(20))  # Show more rows

                # Summary statistics
                disease_count = (preds == 1).sum()
                no_disease_count = (preds == 0).sum()
                st.info(
                    f"📊 Summary: {no_disease_count} predicted as No Disease, "
                    f"{disease_count} predicted as Disease"
                )

                # Download button
                csv_bytes = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv",
                )
                
                logger.info(f"Prediction completed: {disease_count} disease, {no_disease_count} no disease")
            
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.error("Please check that your CSV format matches the training data.")
                logger.error(f"Prediction error: {e}", exc_info=True)

else:
    st.info("Please upload a CSV file from the sidebar to start.")


# Footer info
st.sidebar.markdown("---")
st.sidebar.write(metric_info)
st.sidebar.write("0 = No Disease, 1 = Disease")

