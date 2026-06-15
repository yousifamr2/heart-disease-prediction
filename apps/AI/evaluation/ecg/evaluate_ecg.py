import sys
import os
import json
import ast
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, brier_score_loss, roc_curve, precision_recall_curve,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths
EVAL_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EVAL_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVAL_ROOT))

from config import (
    PTBXL_DATASET_PATH, EVAL_SAMPLE_SIZE, FIGURES_ECG_DIR,
    REPORTS_DIR, SCORECARDS_DIR
)

try:
    from app.services.ecg_service import get_ecg_predictor, _label_for_code
except ImportError as e:
    print(f"Error importing ecg_service: {e}")
    sys.exit(1)

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def evaluate_ecg():
    print("="*50)
    print("Starting ECG Evaluation Framework")
    print("="*50)
    
    if not PTBXL_DATASET_PATH or not Path(PTBXL_DATASET_PATH).exists():
        print(f"Error: PTBXL_DATASET_PATH '{PTBXL_DATASET_PATH}' does not exist or is not set.")
        print("Please download the PTB-XL dataset from PhysioNet (https://physionet.org/content/ptb-xl/)")
        print("Set the environment variable PTBXL_DATASET_PATH to the dataset root directory.")
        print("Example: set PTBXL_DATASET_PATH=C:\\data\\ptb-xl")
        sys.exit(1)
        
    db_path = Path(PTBXL_DATASET_PATH)
    csv_path = db_path / "ptbxl_database.csv"
    
    if not csv_path.exists():
        print(f"Error: ptbxl_database.csv not found in {db_path}")
        sys.exit(1)
        
    print(f"Loading PTB-XL database from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Parse scp_codes dictionary
    df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)
    
    # We want to stratify. To do so simply, we'll pick the top diagnostic class for each record.
    # We only care about diagnostic codes. We can just use the highest probability code.
    def get_main_code(scp_dict):
        if not scp_dict:
            return "NORM"
        return max(scp_dict.items(), key=lambda x: x[1])[0]
        
    df['main_code'] = df['scp_codes'].apply(get_main_code)
    
    # Drop rows where filename_hr is missing if it's there
    if 'filename_hr' in df.columns:
        filename_col = 'filename_hr'
    elif 'filename_lr' in df.columns:
        filename_col = 'filename_lr'
    else:
        print("Error: Could not find filename_hr or filename_lr columns.")
        sys.exit(1)
        
    df = df.dropna(subset=[filename_col])
    
    # Keep only classes that have enough samples for stratification
    class_counts = df['main_code'].value_counts()
    valid_classes = class_counts[class_counts > 10].index
    df = df[df['main_code'].isin(valid_classes)]
    
    sample_size = min(EVAL_SAMPLE_SIZE, len(df))
    print(f"Sampling {sample_size} records using stratified sampling...")
    try:
        sampled_df = df.groupby('main_code', group_keys=False).apply(lambda x: x.sample(min(len(x), max(1, int(sample_size * len(x) / len(df))))))
        if len(sampled_df) < sample_size:
            sampled_df = pd.concat([sampled_df, df.drop(sampled_df.index).sample(sample_size - len(sampled_df))])
    except Exception as e:
        print("Stratified sampling failed, falling back to random sampling.")
        sampled_df = df.sample(sample_size, random_state=42)
        
    sampled_df = sampled_df.reset_index(drop=True)
    
    print("Loading model and predictor...")
    try:
        predictor = get_ecg_predictor()
    except Exception as e:
        print(f"Failed to load ECG Predictor: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    classes = predictor.classes
    num_classes = len(classes)
    
    y_true_all = []
    y_prob_all = []
    
    print("Running inference on samples...")
    for idx, row in sampled_df.iterrows():
        record_path = db_path / row[filename_col]
        try:
            # wfdb.rdsamp expects the path without the .dat or .hea extension
            signal, meta = wfdb.rdsamp(str(record_path))
        except Exception as e:
            print(f"Failed to load record {record_path}: {e}")
            continue
            
        # Target labels
        # Convert scp_codes to binary array matching predictor.classes
        record_codes = row['scp_codes']
        y_t = np.zeros(num_classes)
        for i, cls in enumerate(classes):
            # Threshold for ground truth is usually > 0 in scp_codes dictionary for PTB-XL
            if cls in record_codes and record_codes[cls] > 0:
                y_t[i] = 1
        y_true_all.append(y_t)
        
        # Inference
        try:
            tensor_signal = predictor.preprocess_signal(signal)
            with torch.no_grad():
                logits = predictor.model(tensor_signal)
                probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
            y_prob_all.append(probs)
        except Exception as e:
            print(f"Failed inference on record {record_path}: {e}")
            y_true_all.pop() # Remove the added true label
            continue
            
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(sampled_df)}")
            
    if not y_true_all:
        print("No records processed. Exiting.")
        sys.exit(1)
        
    Y_true = np.array(y_true_all)
    Y_prob = np.array(y_prob_all)
    
    print("Computing metrics...")
    
    # --- Classification Metrics ---
    # Default threshold 0.5
    Y_pred = (Y_prob >= 0.5).astype(int)
    
    macro_f1 = f1_score(Y_true, Y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(Y_true, Y_pred, average='micro', zero_division=0)
    weighted_f1 = f1_score(Y_true, Y_pred, average='weighted', zero_division=0)
    
    try:
        macro_auc = roc_auc_score(Y_true, Y_prob, average='macro')
        micro_auc = roc_auc_score(Y_true, Y_prob, average='micro')
        weighted_auc = roc_auc_score(Y_true, Y_prob, average='weighted')
    except ValueError:
        macro_auc, micro_auc, weighted_auc = 0.0, 0.0, 0.0
        
    # Flat metrics for clinical stats
    Y_true_flat = Y_true.flatten()
    Y_pred_flat = Y_pred.flatten()
    Y_prob_flat = Y_prob.flatten()
    
    accuracy = accuracy_score(Y_true_flat, Y_pred_flat)
    precision = precision_score(Y_true_flat, Y_pred_flat, zero_division=0)
    recall = recall_score(Y_true_flat, Y_pred_flat, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(Y_true_flat, Y_pred_flat, labels=[0,1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    mcc = matthews_corrcoef(Y_true_flat, Y_pred_flat)
    kappa = cohen_kappa_score(Y_true_flat, Y_pred_flat)
    
    # Calibration
    brier = brier_score_loss(Y_true_flat, Y_prob_flat)
    ece = expected_calibration_error(Y_true_flat, Y_prob_flat)
    
    # Identify top 10 classes
    class_frequencies = Y_true.sum(axis=0)
    top_10_idx = np.argsort(class_frequencies)[::-1][:10]
    top_10_classes = [classes[i] for i in top_10_idx]
    
    # --- VISUALIZATIONS ---
    print("Generating visualizations...")
    
    # 1. Confusion Matrix (Top 10 Classes, flattened across them)
    Y_true_top10 = Y_true[:, top_10_idx].flatten()
    Y_pred_top10 = Y_pred[:, top_10_idx].flatten()
    cm = confusion_matrix(Y_true_top10, Y_pred_top10)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix (Top 10 Classes Aggregated)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(FIGURES_ECG_DIR / 'confusion_matrix.png', bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curves (Top 10)
    plt.figure(figsize=(10, 8))
    for i in top_10_idx:
        if len(np.unique(Y_true[:, i])) > 1:
            fpr, tpr, _ = roc_curve(Y_true[:, i], Y_prob[:, i])
            auc_val = roc_auc_score(Y_true[:, i], Y_prob[:, i])
            plt.plot(fpr, tpr, label=f"{classes[i]} (AUC = {auc_val:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Top 10 Classes)')
    plt.legend(loc='lower right')
    plt.savefig(FIGURES_ECG_DIR / 'roc_curves.png', bbox_inches='tight')
    plt.close()
    
    # 3. Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    for i in top_10_idx:
        if len(np.unique(Y_true[:, i])) > 1:
            p, r, _ = precision_recall_curve(Y_true[:, i], Y_prob[:, i])
            plt.plot(r, p, label=f"{classes[i]}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves (Top 10 Classes)')
    plt.legend(loc='lower left')
    plt.savefig(FIGURES_ECG_DIR / 'precision_recall_curves.png', bbox_inches='tight')
    plt.close()
    
    # 4. Class Distribution
    plt.figure(figsize=(12, 6))
    top_freqs = class_frequencies[top_10_idx]
    sns.barplot(x=top_10_classes, y=top_freqs, palette='viridis')
    plt.title('Class Distribution (Top 10 Classes)')
    plt.xlabel('ECG Diagnosis')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.savefig(FIGURES_ECG_DIR / 'class_distribution.png', bbox_inches='tight')
    plt.close()
    
    # 5. Confidence Histogram
    plt.figure(figsize=(10, 6))
    correct = (Y_true_flat == Y_pred_flat)
    incorrect = ~correct
    plt.hist(Y_prob_flat[correct], bins=50, alpha=0.5, color='green', label='Correct')
    plt.hist(Y_prob_flat[incorrect], bins=50, alpha=0.5, color='red', label='Incorrect')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(FIGURES_ECG_DIR / 'confidence_histogram.png', bbox_inches='tight')
    plt.close()
    
    # 6. Reliability Diagram
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(Y_true_flat, Y_prob_flat, n_bins=10)
    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.title('Reliability Diagram (Calibration Curve)')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend()
    plt.savefig(FIGURES_ECG_DIR / 'reliability_diagram.png', bbox_inches='tight')
    plt.close()
    
    # --- Threshold Optimization ---
    print("Performing Threshold Analysis...")
    thresholds = np.linspace(0.05, 0.95, 19)
    f1_scores = []
    recall_scores = []
    precision_scores = []
    
    for th in thresholds:
        yp = (Y_prob_flat >= th).astype(int)
        f1_scores.append(f1_score(Y_true_flat, yp, zero_division=0))
        recall_scores.append(recall_score(Y_true_flat, yp, zero_division=0))
        precision_scores.append(precision_score(Y_true_flat, yp, zero_division=0))
        
    best_f1_idx = np.argmax(f1_scores)
    best_recall_idx = np.argmax(recall_scores)
    
    opt_thresholds = {
        "best_f1_threshold": float(thresholds[best_f1_idx]),
        "best_f1_score": float(f1_scores[best_f1_idx]),
        "best_recall_threshold": float(thresholds[best_recall_idx]),
        "best_recall_score": float(recall_scores[best_recall_idx]),
    }
    
    with open(FIGURES_ECG_DIR / 'optimal_thresholds.json', 'w') as f:
        json.dump(opt_thresholds, f, indent=4)
        
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_scores, marker='o', color='purple')
    plt.axvline(thresholds[best_f1_idx], color='r', linestyle='--', label=f'Best F1 ({thresholds[best_f1_idx]:.2f})')
    plt.title('Threshold vs F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig(FIGURES_ECG_DIR / 'threshold_vs_f1.png', bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, recall_scores, marker='o', color='blue')
    plt.title('Threshold vs Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.savefig(FIGURES_ECG_DIR / 'threshold_vs_recall.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision_scores, marker='o', color='green')
    plt.title('Threshold vs Precision')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.savefig(FIGURES_ECG_DIR / 'threshold_vs_precision.png', bbox_inches='tight')
    plt.close()
    
    # --- ECG Scorecard ---
    print("Generating Scorecard...")
    # Normalize metrics to 0-100
    ecg_score = (macro_auc + macro_f1 + accuracy + sensitivity + specificity + (1 - brier)) / 6.0 * 100
    
    scorecard_data = [
        {"Metric": "Accuracy", "Score": round(accuracy * 100, 2)},
        {"Metric": "F1 (Macro)", "Score": round(macro_f1 * 100, 2)},
        {"Metric": "ROC AUC (Macro)", "Score": round(macro_auc * 100, 2)},
        {"Metric": "Sensitivity", "Score": round(sensitivity * 100, 2)},
        {"Metric": "Specificity", "Score": round(specificity * 100, 2)},
        {"Metric": "Calibration (1-Brier)", "Score": round((1 - brier) * 100, 2)},
        {"Metric": "MCC", "Score": round(((mcc + 1) / 2) * 100, 2)}, # MCC is -1 to 1
    ]
    
    scorecard_df = pd.DataFrame(scorecard_data)
    scorecard_df.to_csv(SCORECARDS_DIR / 'ecg_scorecard.csv', index=False)
    
    with open(SCORECARDS_DIR / 'ecg_overall_score.txt', 'w') as f:
        f.write(str(round(ecg_score, 2)))
        
    print(f"Overall ECG Score: {ecg_score:.2f}/100")
    print("ECG Evaluation Complete.")

if __name__ == "__main__":
    evaluate_ecg()
