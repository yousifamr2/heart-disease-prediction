"""
Generate Synthetic Heart Disease Test Cases
============================================
This script generates synthetic CSV datasets for heart disease prediction testing.
The generated data maintains the same structure as the original dataset but contains
completely new, synthetic samples with corresponding true labels.

Features:
- Automatically extracts feature statistics from original dataset
- Generates realistic synthetic data based on distributions
- Creates 3 risk-level datasets (low, medium, high) with 5 rows each
- Generates true_labels.csv file with corresponding labels
- Validates data consistency and saves to CSV files
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================
ORIGINAL_DATA_PATH = "AI/data/heart_statlog_cleveland_hungary_final.csv"
OUTPUT_DIR = "data/synthetic_test_cases"
NUM_SAMPLES_PER_FILE = 5  # Exactly 5 rows per file
TARGET_COLUMN = "target"  # Column to exclude from output


# ============================================================================
# Data Analysis Functions
# ============================================================================

def load_original_data(path: str) -> pd.DataFrame:
    """
    Load the original dataset.
    
    Parameters:
    -----------
    path : str
        Path to the original CSV file
    
    Returns:
    --------
    pd.DataFrame : Original dataset
    """
    logger.info(f"Loading original dataset from: {path}")
    if not os.path.exists(path):
        logger.error(f"Original dataset not found at: {path}")
        raise FileNotFoundError(f"Original dataset not found at: {path}")
    
    df = pd.read_csv(path, sep=",")
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def analyze_features(df: pd.DataFrame, target_col: str = "target"):
    """
    Analyze features to extract statistics for synthetic generation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataset
    target_col : str
        Name of target column to exclude
    
    Returns:
    --------
    dict : Dictionary with feature analysis results
    """
    logger.info("Analyzing features...")
    
    # Get feature columns (exclude target)
    feature_cols = [col for col in df.columns if col != target_col]
    
    analysis = {
        "feature_names": feature_cols,
        "numerical_features": {},
        "categorical_features": {},
        "data_types": {}
    }
    
    # Analyze each feature
    for col in feature_cols:
        dtype = df[col].dtype
        analysis["data_types"][col] = dtype
        
        # Check if numerical or categorical
        if pd.api.types.is_numeric_dtype(dtype):
            # Numerical feature
            analysis["numerical_features"][col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "median": float(df[col].median())
            }
            logger.info(f"  {col}: Numerical (range: {df[col].min():.2f} - {df[col].max():.2f})")
        else:
            # Categorical feature
            unique_vals = df[col].unique().tolist()
            value_counts = df[col].value_counts().to_dict()
            analysis["categorical_features"][col] = {
                "unique_values": unique_vals,
                "value_counts": value_counts,
                "most_common": df[col].mode().iloc[0] if len(df[col].mode()) > 0 else unique_vals[0]
            }
            logger.info(f"  {col}: Categorical ({len(unique_vals)} unique values)")
    
    logger.info(f"Summary: {len(analysis['numerical_features'])} numerical, "
                f"{len(analysis['categorical_features'])} categorical features")
    
    return analysis


# ============================================================================
# Synthetic Data Generation Functions
# ============================================================================

def generate_numerical_value(feature_name: str, stats: dict, risk_level: str = "medium"):
    """
    Generate a synthetic numerical value based on statistics and risk level.
    
    Parameters:
    -----------
    feature_name : str
        Name of the feature
    stats : dict
        Statistics dictionary (min, max, mean, std, median)
    risk_level : str
        Risk level: "low", "medium", "high"
    
    Returns:
    --------
    float : Generated numerical value
    """
    min_val = stats["min"]
    max_val = stats["max"]
    mean_val = stats["mean"]
    std_val = stats["std"]
    median_val = stats["median"]
    
    # Adjust mean based on risk level for health-related features
    # High risk: higher values for negative indicators (age, cholesterol, bp, etc.)
    # Low risk: lower values for negative indicators
    
    if risk_level == "high":
        # Shift towards higher risk values
        adjusted_mean = mean_val + 0.8 * std_val
    elif risk_level == "low":
        # Shift towards lower risk values
        adjusted_mean = mean_val - 0.8 * std_val
    else:  # medium
        adjusted_mean = mean_val
    
    # Generate value from normal distribution, clipped to valid range
    value = np.random.normal(adjusted_mean, std_val * 0.7)
    
    # Apply logical constraints based on feature name
    feature_lower = feature_name.lower()
    
    if "age" in feature_lower:
        value = np.clip(value, 18, 100)  # Age must be reasonable
        return int(round(value))
    elif "cholesterol" in feature_lower:
        value = np.clip(value, 100, max_val * 1.1)  # Cholesterol must be positive
    elif "bp" in feature_lower or "blood" in feature_lower:
        value = np.clip(value, 70, max_val * 1.1)  # Blood pressure must be reasonable
    elif "heart" in feature_lower or "rate" in feature_lower:
        value = np.clip(value, 40, max_val * 1.1)  # Heart rate must be positive
    elif "oldpeak" in feature_lower:
        value = np.clip(value, -2, max_val * 1.1)  # Oldpeak can be negative
    else:
        value = np.clip(value, max(0, min_val * 0.9), max_val * 1.1)
    
    # Round to appropriate decimal places
    if "age" in feature_lower or "heart" in feature_lower or "bp" in feature_lower:
        return int(round(value))
    else:
        return round(value, 2)


def generate_categorical_value(feature_name: str, stats: dict, risk_level: str = "medium"):
    """
    Generate a synthetic categorical value.
    
    Parameters:
    -----------
    feature_name : str
        Name of the feature
    stats : dict
        Statistics dictionary (unique_values, value_counts, most_common)
    risk_level : str
        Risk level (for weighted sampling if applicable)
    
    Returns:
    --------
    Any : Generated categorical value
    """
    unique_vals = stats["unique_values"]
    
    # For binary or low-cardinality features, sample uniformly
    if len(unique_vals) <= 5:
        return np.random.choice(unique_vals)
    else:
        # For high-cardinality, use weighted sampling based on frequency
        values = list(stats["value_counts"].keys())
        weights = list(stats["value_counts"].values())
        weights = np.array(weights) / sum(weights)  # Normalize
        return np.random.choice(values, p=weights)


def generate_synthetic_dataset(analysis: dict, n_samples: int, risk_level: str = "medium"):
    """
    Generate a complete synthetic dataset.
    
    Parameters:
    -----------
    analysis : dict
        Feature analysis dictionary
    n_samples : int
        Number of samples to generate
    risk_level : str
        Risk level: "low", "medium", "high"
    
    Returns:
    --------
    pd.DataFrame : Generated synthetic dataset
    """
    logger.info(f"Generating {n_samples} samples for {risk_level} risk level...")
    
    data = {}
    feature_names = analysis["feature_names"]
    
    # Generate values for each feature
    for col in feature_names:
        if col in analysis["numerical_features"]:
            # Generate numerical values
            stats = analysis["numerical_features"][col]
            data[col] = [
                generate_numerical_value(col, stats, risk_level)
                for _ in range(n_samples)
            ]
        elif col in analysis["categorical_features"]:
            # Generate categorical values - validate they're from valid set
            stats = analysis["categorical_features"][col]
            valid_values = stats["unique_values"]
            data[col] = []
            for _ in range(n_samples):
                value = generate_categorical_value(col, stats, risk_level)
                # Strict validation: ensure value is in original dataset
                if value not in valid_values:
                    logger.warning(f"Generated invalid categorical value for {col}: {value}. Using most common.")
                    value = stats["most_common"]
                data[col].append(value)
        else:
            # Fallback: use median/mode
            if col in analysis["numerical_features"]:
                data[col] = [analysis["numerical_features"][col]["median"]] * n_samples
            else:
                data[col] = [analysis["categorical_features"][col]["most_common"]] * n_samples
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure correct column order
    df = df[feature_names]
    
    # Ensure correct data types with safe casting
    for col in feature_names:
        if col in analysis["data_types"]:
            try:
                df[col] = df[col].astype(analysis["data_types"][col])
            except Exception as e:
                logger.warning(f"Could not cast column {col} to {analysis['data_types'][col]} — leaving generated dtype. Error: {e}")
    
    # Shuffle rows to ensure randomness
    df = df.sample(frac=1, random_state=np.random.randint(0, 10000)).reset_index(drop=True)
    
    logger.info(f"Generated {len(df)} rows with {len(df.columns)} features")
    
    return df


def generate_true_labels(risk_levels: list, samples_per_file: int):
    """
    Generate true labels for all test cases.
    
    Parameters:
    -----------
    risk_levels : list
        List of risk level names
    samples_per_file : int
        Number of samples per file
    
    Returns:
    --------
    pd.DataFrame : DataFrame with sample_id and true_target columns
    
    Note:
    -----
    sample_id increments across all files (1..15 for 3 files × 5 rows each)
    """
    labels_data = []
    sample_id = 1  # Start from 1 and increment across all files
    
    for risk_level in risk_levels:
        # Determine label distribution based on risk level
        if risk_level == "low_risk":
            # Low risk: mostly 0 (no disease) - 80% label 0, 20% label 1
            label_probs = [0.8, 0.2]
        elif risk_level == "high_risk":
            # High risk: mostly 1 (disease) - 20% label 0, 80% label 1
            label_probs = [0.2, 0.8]
        else:  # medium_risk
            # Medium risk: mixed distribution - 50% label 0, 50% label 1
            label_probs = [0.5, 0.5]
        
        # Generate labels for this risk level
        for _ in range(samples_per_file):
            true_target = np.random.choice([0, 1], p=label_probs)
            labels_data.append({
                "sample_id": sample_id,
                "true_target": true_target
            })
            sample_id += 1
    
    logger.info(f"Generated {len(labels_data)} labels with sample_id 1..{sample_id-1}")
    return pd.DataFrame(labels_data)


def ensure_no_duplicates(all_dataframes: list, analysis: dict, risk_levels: list, max_attempts: int = 5) -> list:
    """
    Ensure no duplicate rows across all generated datasets.
    Implements actual duplicate detection and regeneration.
    
    Parameters:
    -----------
    all_dataframes : list
        List of DataFrames to check
    analysis : dict
        Feature analysis dictionary for regeneration
    risk_levels : list
        List of risk level names
    max_attempts : int
        Maximum attempts to regenerate duplicates (default: 5)
    
    Returns:
    --------
    list : List of DataFrames with duplicates removed/regenerated
    """
    # Combine all dataframes to check for duplicates
    combined = pd.concat(all_dataframes, ignore_index=True)
    
    # Check for duplicates
    duplicates_mask = combined.duplicated(keep=False)
    
    if duplicates_mask.any():
        num_duplicates = duplicates_mask.sum()
        logger.warning(f"Found {num_duplicates} duplicate rows across datasets. Attempting to regenerate...")
        
        # Find indices of duplicates
        duplicate_indices = combined[duplicates_mask].index.tolist()
        
        attempts = 0
        while attempts < max_attempts and combined.duplicated(keep=False).any():
            attempts += 1
            logger.info(f"Deduplication attempt {attempts}/{max_attempts}")
            
            # For each duplicate, regenerate that row
            for idx in duplicate_indices:
                # Determine which dataframe and row this belongs to
                df_idx = idx // len(all_dataframes[0])
                row_idx = idx % len(all_dataframes[0])
                
                if df_idx < len(all_dataframes):
                    risk_level = risk_levels[df_idx].replace("_risk", "")
                    
                    # Regenerate this single row
                    new_row = {}
                    for col in analysis["feature_names"]:
                        if col in analysis["numerical_features"]:
                            stats = analysis["numerical_features"][col]
                            new_row[col] = generate_numerical_value(col, stats, risk_level)
                        elif col in analysis["categorical_features"]:
                            stats = analysis["categorical_features"][col]
                            new_row[col] = generate_categorical_value(col, stats, risk_level)
                    
                    # Update the row in the corresponding dataframe
                    for col, value in new_row.items():
                        all_dataframes[df_idx].at[row_idx, col] = value
            
            # Recombine and check
            combined = pd.concat(all_dataframes, ignore_index=True)
            duplicate_indices = combined[combined.duplicated(keep=False)].index.tolist()
            
            if not combined.duplicated(keep=False).any():
                logger.info("All duplicates resolved successfully")
                break
        
        if combined.duplicated(keep=False).any():
            logger.warning(f"Unable to resolve all duplicates after {max_attempts} attempts. Some duplicates may remain.")
    else:
        logger.info("No duplicates found across datasets")
    
    return all_dataframes


def validate_generated_data(df: pd.DataFrame, analysis: dict) -> tuple:
    """
    Validate generated data matches original structure.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Generated dataset
    analysis : dict
        Feature analysis dictionary
    
    Returns:
    --------
    tuple : (is_valid, issues_list)
    """
    issues = []
    
    # Check column names
    expected_cols = set(analysis["feature_names"])
    actual_cols = set(df.columns)
    
    if expected_cols != actual_cols:
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols
        if missing:
            issues.append(f"Missing columns: {missing}")
        if extra:
            issues.append(f"Extra columns: {extra}")
    
    # Check column order
    if list(df.columns) != analysis["feature_names"]:
        issues.append("Column order mismatch")
    
    # Check data types
    for col in analysis["feature_names"]:
        if col in df.columns:
            expected_dtype = analysis["data_types"][col]
            actual_dtype = df[col].dtype
            if expected_dtype != actual_dtype:
                issues.append(f"Data type mismatch for {col}: expected {expected_dtype}, got {actual_dtype}")
    
    # Check value ranges for numerical features
    for col in analysis["numerical_features"]:
        if col in df.columns:
            stats = analysis["numerical_features"][col]
            min_val = df[col].min()
            max_val = df[col].max()
            if min_val < stats["min"] * 0.7 or max_val > stats["max"] * 1.3:
                issues.append(f"Value range issue for {col}: values outside expected range")
    
    # Check categorical values
    for col in analysis["categorical_features"]:
        if col in df.columns:
            valid_values = set(analysis["categorical_features"][col]["unique_values"])
            actual_values = set(df[col].unique())
            invalid = actual_values - valid_values
            if invalid:
                issues.append(f"Invalid categorical values for {col}: {invalid}")
    
    is_valid = len(issues) == 0
    return is_valid, issues


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main function to generate synthetic datasets."""
    logger.info("=" * 70)
    logger.info("Synthetic Heart Disease Test Cases Generator")
    logger.info("=" * 70)
    
    # Step 1: Load original data - gracefully exit if missing
    try:
        original_df = load_original_data(ORIGINAL_DATA_PATH)
    except Exception as e:
        logger.error(f"Failed to load original dataset: {e}")
        return
    
    # Step 2: Analyze features
    analysis = analyze_features(original_df, target_col=TARGET_COLUMN)
    
    # Step 3: Create output directory (platform-independent path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Step 4: Generate datasets for different risk levels
    risk_levels = ["low_risk", "medium_risk", "high_risk"]
    generated_files = []
    all_dataframes = []
    
    for risk_level in risk_levels:
        # Generate dataset
        synthetic_df = generate_synthetic_dataset(
            analysis=analysis,
            n_samples=NUM_SAMPLES_PER_FILE,
            risk_level=risk_level.replace("_risk", "")
        )
        
        # Validate
        is_valid, issues = validate_generated_data(synthetic_df, analysis)
        
        # Save to CSV using platform-independent path
        filename = f"test_cases_{risk_level}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        synthetic_df.to_csv(filepath, index=False)
        generated_files.append((filename, len(synthetic_df), is_valid, issues))
        all_dataframes.append(synthetic_df)
        
        logger.info(f"Saved: {filepath}")
        if not is_valid:
            logger.warning(f"Validation issues found for {filename}: {issues}")
    
    # Step 5: Ensure no duplicates across files
    logger.info("Checking for duplicate rows across files...")
    all_dataframes = ensure_no_duplicates(all_dataframes, analysis, risk_levels)
    
    # Step 6: Generate true labels file
    logger.info("Generating true labels file...")
    labels_df = generate_true_labels(risk_levels, NUM_SAMPLES_PER_FILE)
    labels_filepath = os.path.join(OUTPUT_DIR, "true_labels.csv")
    labels_df.to_csv(labels_filepath, index=False)
    logger.info(f"Saved: {labels_filepath}")
    
    # Step 7: Validate label alignment with assertions
    total_samples = len(risk_levels) * NUM_SAMPLES_PER_FILE
    label_alignment_ok = len(labels_df) == total_samples
    
    # Critical validation assertions
    try:
        assert total_samples == 3 * NUM_SAMPLES_PER_FILE, \
            f"Expected total_samples={3 * NUM_SAMPLES_PER_FILE}, got {total_samples}"
        assert len(labels_df) == total_samples, \
            f"Label count mismatch: expected {total_samples}, got {len(labels_df)}"
        logger.info("✓ Validation assertions passed")
    except AssertionError as e:
        logger.error(f"✗ Validation assertion failed: {e}")
        raise
    
    # Step 8: Print summary
    logger.info("\n" + "=" * 70)
    logger.info("GENERATION SUMMARY")
    logger.info("=" * 70)
    print(f"{'File':<45} {'Rows':<10} {'Status':<10}")
    print("-" * 70)
    
    for filename, n_rows, is_valid, issues in generated_files:
        status = "PASSED" if is_valid else "FAILED"
        print(f"{filename:<45} {n_rows:<10} {status:<10}")
        if issues:
            for issue in issues:
                logger.warning(f"  - {issue}")
    
    # Feature schema check
    print("\n" + "-" * 70)
    print("FEATURE SCHEMA CHECK")
    print("-" * 70)
    schema_ok = all(status == "PASSED" for _, _, status, _ in [(f, r, "PASSED" if v else "FAILED", i) for f, r, v, i in generated_files])
    print(f"Schema validation: {'PASSED' if schema_ok else 'FAILED'}")
    
    # Label alignment check
    print("\n" + "-" * 70)
    print("LABEL ALIGNMENT CHECK")
    print("-" * 70)
    print(f"Total samples: {total_samples}")
    print(f"Total labels: {len(labels_df)}")
    print(f"Alignment: {'PASSED' if label_alignment_ok else 'FAILED'}")
    
    # Label distribution summary
    print("\n" + "-" * 70)
    print("LABEL DISTRIBUTION")
    print("-" * 70)
    for risk_level in risk_levels:
        start_idx = risk_levels.index(risk_level) * NUM_SAMPLES_PER_FILE
        end_idx = start_idx + NUM_SAMPLES_PER_FILE
        risk_labels = labels_df.iloc[start_idx:end_idx]["true_target"]
        label_0_count = (risk_labels == 0).sum()
        label_1_count = (risk_labels == 1).sum()
        logger.info(f"{risk_level}: {label_0_count} samples with label 0, {label_1_count} samples with label 1")
    
    print("=" * 70)
    logger.info(f"All files saved to: {OUTPUT_DIR}")
    logger.info("Generation complete!")


if __name__ == "__main__":
    # Set random seed for reproducibility (preserve existing seed)
    np.random.seed(42)
    logger.info("Random seed set to 42 for reproducibility")
    main()
