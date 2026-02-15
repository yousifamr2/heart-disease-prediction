"""
Validate Model Predictions on Synthetic Test Data
==================================================
This script loads the synthetic test cases, makes predictions using the trained model,
and compares them with the true labels to assess model accuracy.
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Paths
TEST_CASES_DIR = "data/synthetic_test_cases"
TRUE_LABELS_PATH = os.path.join(TEST_CASES_DIR, "true_labels.csv")
MODEL_PATHS = [
    "AI/models/best_model.pkl",
    "models/best_model.pkl",
    "./AI/models/best_model.pkl",
    "./models/best_model.pkl"
]


def load_test_cases():
    """Load all synthetic test case files and combine them."""
    risk_levels = ["low_risk", "medium_risk", "high_risk"]
    all_data = []
    
    for risk_level in risk_levels:
        filepath = os.path.join(TEST_CASES_DIR, f"test_cases_{risk_level}.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['risk_level'] = risk_level
            all_data.append(df)
            logger.info(f"Loaded {len(df)} samples from {risk_level}")
        else:
            logger.warning(f"File not found: {filepath}")
    
    if not all_data:
        raise FileNotFoundError("No test case files found!")
    
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total test cases loaded: {len(combined)}")
    return combined


def load_true_labels():
    """Load the true labels CSV file."""
    if not os.path.exists(TRUE_LABELS_PATH):
        raise FileNotFoundError(f"True labels file not found: {TRUE_LABELS_PATH}")
    
    labels_df = pd.read_csv(TRUE_LABELS_PATH)
    logger.info(f"Loaded {len(labels_df)} true labels")
    return labels_df


def load_model():
    """Load the trained model from disk."""
    model_path = None
    for path in MODEL_PATHS:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        raise FileNotFoundError("Model file not found in any expected location!")
    
    logger.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
    return model, model_path


def make_predictions(model, X_test, threshold=0.7):
    """
    Make predictions using the loaded model with custom threshold.
    
    Parameters:
    -----------
    model : trained model
        The trained model to use for predictions
    X_test : pd.DataFrame
        Test features
    threshold : float
        Probability threshold for classification (default: 0.7)
        If probability of Disease >= threshold, predict Disease (1)
        Otherwise, predict No Disease (0)
    """
    try:
        # Check if model has feature_names_in_ to ensure correct column order
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            # Reorder columns to match model's expected order
            missing_cols = set(expected_features) - set(X_test.columns)
            if missing_cols:
                logger.error(f"Missing columns: {missing_cols}")
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            X_test = X_test[expected_features]
            logger.info("Reordered columns to match model's expected features")
        
        # Get probabilities first
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(X_test)
                logger.info("Generated predictions with probabilities")
                
                # Apply custom threshold (70%)
                # probabilities[:, 1] is the probability of Disease class
                if probabilities.shape[1] >= 2:
                    disease_proba = probabilities[:, 1]
                    predictions = (disease_proba >= threshold).astype(int)
                    logger.info(f"Applied threshold {threshold*100:.0f}%: {predictions.sum()} predicted as Disease")
                else:
                    # Fallback to default predict if probabilities shape is unexpected
                    predictions = model.predict(X_test)
                    logger.warning("Using default predict() due to unexpected probability shape")
            except Exception as e:
                logger.warning(f"Could not generate probabilities: {e}, using default predict()")
                predictions = model.predict(X_test)
        else:
            # If model doesn't support predict_proba, use default predict
            predictions = model.predict(X_test)
            logger.warning("Model does not support predict_proba, using default threshold (50%)")
        
        return predictions, probabilities
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def validate_predictions(y_true, y_pred, X_test, risk_levels):
    """Compare predictions with true labels and calculate metrics."""
    # Calculate overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=['No Disease', 'Disease'])
    
    # Create detailed comparison DataFrame
    comparison_df = pd.DataFrame({
        'sample_id': range(1, len(y_true) + 1),
        'risk_level': risk_levels,
        'true_label': y_true,
        'predicted_label': y_pred,
        'correct': y_true == y_pred
    })
    
    # Add label names for readability
    comparison_df['true_label_name'] = comparison_df['true_label'].map({0: 'No Disease', 1: 'Disease'})
    comparison_df['predicted_label_name'] = comparison_df['predicted_label'].map({0: 'No Disease', 1: 'Disease'})
    
    return accuracy, cm, report, comparison_df


def print_results(accuracy, cm, report, comparison_df, threshold=0.7):
    """Print formatted results."""
    print("\n" + "="*80)
    print(f"MODEL PREDICTION VALIDATION RESULTS (Threshold: {threshold*100:.0f}%)")
    print("="*80)
    
    print(f"\nOverall Accuracy: {accuracy:.2%} ({accuracy*100:.2f}%)")
    print(f"   Correct Predictions: {(comparison_df['correct']).sum()} / {len(comparison_df)}")
    print(f"   Incorrect Predictions: {(~comparison_df['correct']).sum()} / {len(comparison_df)}")
    
    print("\n" + "-"*80)
    print("CONFUSION MATRIX")
    print("-"*80)
    print("                 Predicted")
    print("              No Disease  Disease")
    print(f"Actual No Disease    {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"       Disease       {cm[1,0]:4d}      {cm[1,1]:4d}")
    
    print("\n" + "-"*80)
    print("CLASSIFICATION REPORT")
    print("-"*80)
    print(report)
    
    print("\n" + "-"*80)
    print("DETAILED PREDICTION COMPARISON")
    print("-"*80)
    display_cols = ['sample_id', 'risk_level', 'true_label_name', 'predicted_label_name', 'correct']
    print(comparison_df[display_cols].to_string(index=False))
    
    # Accuracy by risk level
    print("\n" + "-"*80)
    print("ACCURACY BY RISK LEVEL")
    print("-"*80)
    for risk_level in comparison_df['risk_level'].unique():
        risk_df = comparison_df[comparison_df['risk_level'] == risk_level]
        risk_accuracy = risk_df['correct'].mean()
        print(f"{risk_level:15s}: {risk_accuracy:.2%} ({risk_df['correct'].sum()}/{len(risk_df)} correct)")
    
    # Error analysis
    errors = comparison_df[~comparison_df['correct']]
    if len(errors) > 0:
        print("\n" + "-"*80)
        print("ERROR ANALYSIS")
        print("-"*80)
        print(f"Total Errors: {len(errors)}")
        print("\nError Details:")
        for _, row in errors.iterrows():
            print(f"  Sample {row['sample_id']:2d} ({row['risk_level']:12s}): "
                  f"True={row['true_label_name']:12s}, Predicted={row['predicted_label_name']:12s}")
    
    print("\n" + "="*80)


def main():
    """Main validation function."""
    try:
        # Load data
        logger.info("Loading test cases...")
        test_cases = load_test_cases()
        
        logger.info("Loading true labels...")
        true_labels_df = load_true_labels()
        
        # Ensure same number of samples
        if len(test_cases) != len(true_labels_df):
            raise ValueError(f"Mismatch: {len(test_cases)} test cases but {len(true_labels_df)} labels")
        
        # Extract features (exclude risk_level column)
        X_test = test_cases.drop(columns=['risk_level'], errors='ignore')
        y_true = true_labels_df['true_target'].values
        risk_levels = test_cases['risk_level'].values
        
        # Load model
        logger.info("Loading trained model...")
        model, model_path = load_model()
        
        # Make predictions with 70% threshold
        logger.info("Making predictions with 70% threshold...")
        y_pred, y_proba = make_predictions(model, X_test, threshold=0.7)
        
        # Validate
        logger.info("Validating predictions...")
        accuracy, cm, report, comparison_df = validate_predictions(y_true, y_pred, X_test, risk_levels)
        
        # Print results
        print_results(accuracy, cm, report, comparison_df, threshold=0.7)
        
        # Save detailed results
        output_path = os.path.join(TEST_CASES_DIR, "prediction_results.csv")
        comparison_df.to_csv(output_path, index=False)
        logger.info(f"\nDetailed results saved to: {output_path}")
        
        return accuracy, comparison_df
    
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

