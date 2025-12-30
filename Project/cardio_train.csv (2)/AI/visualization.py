"""
Professional ML Visualization Module
====================================
A comprehensive visualization pipeline for ML classification projects.

This module provides:
1. Data Understanding visualizations
2. Model Evaluation metrics and plots
3. Model Comparison charts
4. Model Explainability (Feature Importance, SHAP)
5. Error Analysis tools

Author: ML Engineering Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Optional SHAP import
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[INFO] SHAP not available. Install with: pip install shap")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


# ============================================================================
# 1. DATA UNDERSTANDING
# ============================================================================

def plot_target_distribution(
    y: pd.Series,
    title: str = "Target/Class Distribution",
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plot target/class distribution with count plot and percentages.
    
    Parameters:
    -----------
    y : pd.Series
        Target variable
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Count plot
    sns.countplot(data=pd.DataFrame({'target': y}), x='target', ax=axes[0], palette='viridis')
    axes[0].set_title(f'{title} - Count', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    
    # Add count labels on bars
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%d')
    
    # Percentage pie chart
    value_counts = y.value_counts()
    colors = sns.color_palette('viridis', len(value_counts))
    axes[1].pie(
        value_counts.values,
        labels=[f'Class {idx}\n({val} samples)' for idx, val in zip(value_counts.index, value_counts.values)],
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )
    axes[1].set_title(f'{title} - Percentage', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_feature_distributions(
    X: pd.DataFrame,
    numerical_cols: Optional[List[str]] = None,
    max_features: int = 12,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Plot histograms and boxplots for numerical features.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature dataframe
    numerical_cols : list, optional
        List of numerical column names. If None, auto-detect.
    max_features : int
        Maximum number of features to plot
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    if numerical_cols is None:
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Limit number of features
    numerical_cols = numerical_cols[:max_features]
    n_features = len(numerical_cols)
    
    if n_features == 0:
        print("[WARN] No numerical features found.")
        return
    
    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, col in enumerate(numerical_cols):
        # Histogram
        hist_idx = idx * 2
        if hist_idx < len(axes):
            axes[hist_idx].hist(X[col].dropna(), bins=30, edgecolor='black', alpha=0.7, color='skyblue')
            axes[hist_idx].set_title(f'{col} - Distribution', fontweight='bold')
            axes[hist_idx].set_xlabel(col)
            axes[hist_idx].set_ylabel('Frequency')
            axes[hist_idx].grid(True, alpha=0.3)
        
        # Boxplot
        box_idx = idx * 2 + 1
        if box_idx < len(axes):
            axes[box_idx].boxplot(X[col].dropna(), vert=True)
            axes[box_idx].set_title(f'{col} - Boxplot', fontweight='bold')
            axes[box_idx].set_ylabel(col)
            axes[box_idx].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features * 2, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_correlation_heatmap(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    numerical_cols: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Plot correlation heatmap for numerical features.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature dataframe
    y : pd.Series, optional
        Target variable (will be included in correlation)
    numerical_cols : list, optional
        List of numerical column names. If None, auto-detect.
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    if numerical_cols is None:
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Prepare data
    data = X[numerical_cols].copy()
    if y is not None:
        data['target'] = y
    
    # Calculate correlation
    corr_matrix = data.corr()
    
    # Plot
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8}
    )
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


# ============================================================================
# 2. MODEL EVALUATION
# ============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix (both raw and normalized).
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : list, optional
        Class names for labels
    normalize : bool
        Whether to normalize the confusion matrix
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Raw confusion matrix
    cm_raw = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm_raw,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names if class_names else ['Class 0', 'Class 1'],
        yticklabels=class_names if class_names else ['Class 0', 'Class 1'],
        ax=axes[0]
    )
    axes[0].set_title('Confusion Matrix (Raw Counts)', fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Normalized confusion matrix
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names if class_names else ['Class 0', 'Class 1'],
        yticklabels=class_names if class_names else ['Class 0', 'Class 1'],
        ax=axes[1]
    )
    axes[1].set_title('Confusion Matrix (Normalized)', fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot ROC curve with AUC score.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels
    y_proba : np.ndarray
        Predicted probabilities for positive class
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    # Handle binary classification
    if y_proba.ndim > 1 and y_proba.shape[1] > 1:
        y_proba = y_proba[:, 1]  # Use positive class probabilities
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return roc_auc


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Precision-Recall Curve",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot Precision-Recall curve.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True binary labels
    y_proba : np.ndarray
        Predicted probabilities for positive class
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    # Handle binary classification
    if y_proba.ndim > 1 and y_proba.shape[1] > 1:
        y_proba = y_proba[:, 1]  # Use positive class probabilities
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='darkblue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return avg_precision


def plot_model_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Model",
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Comprehensive model evaluation visualization.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : np.ndarray
        Predicted probabilities
    model_name : str
        Name of the model
    class_names : list, optional
        Class names
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    dict : Dictionary with evaluation metrics
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Handle probabilities for binary classification
    if y_proba.ndim > 1 and y_proba.shape[1] > 1:
        y_proba_binary = y_proba[:, 1]
    else:
        y_proba_binary = y_proba
    
    # 1. Confusion Matrix (Raw)
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax1,
        xticklabels=class_names if class_names else ['Class 0', 'Class 1'],
        yticklabels=class_names if class_names else ['Class 0', 'Class 1']
    )
    ax1.set_title('Confusion Matrix', fontweight='bold')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # 2. Confusion Matrix (Normalized)
    ax2 = fig.add_subplot(gs[0, 1])
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        ax=ax2,
        xticklabels=class_names if class_names else ['Class 0', 'Class 1'],
        yticklabels=class_names if class_names else ['Class 0', 'Class 1']
    )
    ax2.set_title('Confusion Matrix (Normalized)', fontweight='bold')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    # 3. ROC Curve
    ax3 = fig.add_subplot(gs[0, 2])
    fpr, tpr, _ = roc_curve(y_true, y_proba_binary)
    roc_auc = auc(fpr, tpr)
    ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve', fontweight='bold')
    ax3.legend(loc="lower right")
    ax3.grid(True, alpha=0.3)
    
    # 4. Precision-Recall Curve
    ax4 = fig.add_subplot(gs[1, 0])
    precision, recall, _ = precision_recall_curve(y_true, y_proba_binary)
    avg_precision = average_precision_score(y_true, y_proba_binary)
    ax4.plot(recall, precision, color='darkblue', lw=2, label=f'AP = {avg_precision:.3f}')
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('Precision-Recall Curve', fontweight='bold')
    ax4.legend(loc="lower left")
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    
    # 5. Metrics Summary
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    metrics_text = f"""
    Model: {model_name}
    
    Accuracy:  {acc:.4f}
    Precision: {prec:.4f}
    Recall:    {rec:.4f}
    F1-Score:   {f1:.4f}
    ROC-AUC:    {roc_auc:.4f}
    PR-AUC:     {avg_precision:.4f}
    """
    ax5.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 6. Classification Report (as text)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    report = classification_report(y_true, y_pred, target_names=class_names)
    ax6.text(0.1, 0.5, report, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle(f'{model_name} - Comprehensive Evaluation', fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': avg_precision
    }


# ============================================================================
# 3. MODEL COMPARISON
# ============================================================================

def create_model_comparison_table(
    models_results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a comparison table of models with metrics.
    
    Parameters:
    -----------
    models_results : dict
        Dictionary with model names as keys and metrics dict as values
        Example: {'Model1': {'accuracy': 0.95, 'precision': 0.94, ...}, ...}
    metrics : list
        List of metric names to include
    save_path : str, optional
        Path to save the table as CSV
    
    Returns:
    --------
    pd.DataFrame : Comparison table
    """
    # Prepare data
    data = []
    for model_name, model_metrics in models_results.items():
        row = {'Model': model_name}
        for metric in metrics:
            row[metric.capitalize()] = model_metrics.get(metric, np.nan)
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.set_index('Model')
    
    # Sort by accuracy (or first metric) descending
    if metrics and metrics[0] in df.columns:
        df = df.sort_values(by=metrics[0].capitalize(), ascending=False)
    
    # Display
    print("\n" + "="*60)
    print("MODEL COMPARISON TABLE")
    print("="*60)
    print(df.to_string())
    print("="*60 + "\n")
    
    if save_path:
        df.to_csv(save_path)
        print(f"[INFO] Comparison table saved to {save_path}")
    
    return df


def create_formatted_results_table(
    models_results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
    primary_metric: str = 'accuracy',
    save_path: Optional[str] = None,
    display_style: bool = True
) -> pd.DataFrame:
    """
    Create a comprehensive and formatted results table with all metrics and percentages.
    Similar to Jupyter notebook styled tables.
    
    Parameters:
    -----------
    models_results : dict
        Dictionary with model names as keys and metrics dict as values
        Example: {'Model1': {'accuracy': 0.95, 'precision': 0.94, ...}, ...}
    metrics : list
        List of metric names to include
    primary_metric : str
        Primary metric to sort by (default: 'accuracy')
    save_path : str, optional
        Path to save the table as CSV
    display_style : bool
        Whether to display with formatted output
    
    Returns:
    --------
    pd.DataFrame : Formatted results table
    """
    # Prepare data
    data = []
    for model_name, model_metrics in models_results.items():
        row = {'Model': model_name}
        for metric in metrics:
            metric_key = metric.lower()
            metric_value = model_metrics.get(metric_key, np.nan)
            row[metric.capitalize()] = metric_value
        data.append(row)
    
    # Create DataFrame
    results_table = pd.DataFrame(data)
    
    # Sort by primary metric (descending)
    if primary_metric.lower() in [m.lower() for m in metrics]:
        primary_col = primary_metric.capitalize()
        if primary_col in results_table.columns:
            results_table = results_table.sort_values(by=primary_col, ascending=False).reset_index(drop=True)
    
    # Add percentage columns for better readability
    for metric in metrics:
        metric_col = metric.capitalize()
        if metric_col in results_table.columns:
            percentage_col = f"{metric_col} (%)"
            results_table[percentage_col] = (results_table[metric_col] * 100).round(2)
    
    # Reorder columns: Model, then each metric with its percentage
    column_order = ['Model']
    for metric in metrics:
        metric_col = metric.capitalize()
        if metric_col in results_table.columns:
            column_order.append(metric_col)
            percentage_col = f"{metric_col} (%)"
            if percentage_col in results_table.columns:
                column_order.append(percentage_col)
    
    results_table = results_table[column_order]
    
    # Display formatted table
    if display_style:
        print("\n" + "=" * 90)
        print(" " * 20 + "MACHINE LEARNING MODELS RESULTS COMPARISON")
        print("=" * 90)
        print("\n")
        print(results_table.to_string(index=False))
        print("\n")
        print("=" * 90)
        if primary_metric.lower() in [m.lower() for m in metrics]:
            primary_col = primary_metric.capitalize()
            if primary_col in results_table.columns:
                best_model = results_table.iloc[0]
                best_metric_value = best_model[primary_col]
                best_metric_pct = best_model.get(f"{primary_col} (%)", best_metric_value * 100)
                print(f"\nBest Model: {best_model['Model']} with {best_metric_pct:.2f}% {primary_metric.capitalize()}")
        print("=" * 90 + "\n")
    
    # Save to CSV if path provided
    if save_path:
        results_table.to_csv(save_path, index=False)
        print(f"[INFO] Formatted results table saved to {save_path}")
    
    return results_table


def plot_model_comparison(
    models_results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Bar chart comparing model performance.
    
    Parameters:
    -----------
    models_results : dict
        Dictionary with model names as keys and metrics dict as values
    metrics : list
        List of metric names to plot
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    # Prepare data
    model_names = list(models_results.keys())
    metric_values = {metric: [models_results[model].get(metric, 0) for model in model_names] 
                     for metric in metrics}
    
    # Create grouped bar chart
    x = np.arange(len(model_names))
    width = 0.8 / len(metrics)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics)/2 + 0.5) * width
        ax.bar(x + offset, metric_values[metric], width, label=metric.capitalize(), alpha=0.8)
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics)/2 + 0.5) * width
        for j, val in enumerate(metric_values[metric]):
            ax.text(j + offset, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_training_time_vs_performance(
    models_results: Dict[str, Dict[str, float]],
    training_times: Optional[Dict[str, float]] = None,
    performance_metric: str = 'accuracy',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Scatter plot of training time vs performance.
    
    Parameters:
    -----------
    models_results : dict
        Dictionary with model names as keys and metrics dict as values
    training_times : dict, optional
        Dictionary with model names as keys and training times (seconds) as values
    performance_metric : str
        Metric to use for performance (default: 'accuracy')
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    if training_times is None:
        print("[WARN] Training times not provided. Skipping plot.")
        return
    
    # Prepare data
    model_names = list(models_results.keys())
    performances = [models_results[model].get(performance_metric, 0) for model in model_names]
    times = [training_times.get(model, 0) for model in model_names]
    
    plt.figure(figsize=figsize)
    scatter = plt.scatter(times, performances, s=200, alpha=0.6, c=performances, cmap='viridis')
    
    # Add labels
    for i, name in enumerate(model_names):
        plt.annotate(name, (times[i], performances[i]), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Training Time (seconds)', fontsize=12)
    plt.ylabel(f'{performance_metric.capitalize()}', fontsize=12)
    plt.title('Training Time vs Performance', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label=performance_metric.capitalize())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


# ============================================================================
# 4. MODEL EXPLAINABILITY
# ============================================================================

def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 15,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Plot feature importance for tree-based models.
    
    Parameters:
    -----------
    model : sklearn model
        Trained tree-based model (RandomForest, XGBoost, etc.)
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to display
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    pd.DataFrame : Feature importance dataframe
    """
    # Extract feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models, use absolute coefficients
        importances = np.abs(model.coef_[0])
    else:
        print("[WARN] Model does not support feature importance extraction.")
        return pd.DataFrame()
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=figsize)
    sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return importance_df


def plot_shap_summary(
    model: Any,
    X: pd.DataFrame,
    max_display: int = 15,
    plot_type: str = 'bar',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Plot SHAP summary plot (optional but preferred).
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X : pd.DataFrame
        Feature dataframe (sample for explanation)
    max_display : int
        Maximum number of features to display
    plot_type : str
        Type of SHAP plot ('bar', 'dot', 'violin')
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    shap.Explainer or None
    """
    if not SHAP_AVAILABLE:
        print("[WARN] SHAP is not available. Install with: pip install shap")
        return None
    
    try:
        # Create SHAP explainer
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        
        # Plot
        plt.figure(figsize=figsize)
        if plot_type == 'bar':
            shap.plots.bar(shap_values, max_display=max_display, show=False)
        elif plot_type == 'dot':
            shap.plots.dot(shap_values, max_display=max_display, show=False)
        else:
            shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()
        
        return explainer
    except Exception as e:
        print(f"[ERROR] SHAP plot failed: {e}")
        return None


# ============================================================================
# 5. ERROR ANALYSIS
# ============================================================================

def identify_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: pd.DataFrame,
    return_indices: bool = True
) -> Dict[str, Any]:
    """
    Identify False Positives and False Negatives.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    X : pd.DataFrame
        Feature dataframe
    return_indices : bool
        Whether to return indices of errors
    
    Returns:
    --------
    dict : Dictionary with error information
    """
    # Identify errors
    fp_mask = (y_true == 0) & (y_pred == 1)  # False Positives
    fn_mask = (y_true == 1) & (y_pred == 0)  # False Negatives
    
    fp_indices = np.where(fp_mask)[0] if return_indices else None
    fn_indices = np.where(fn_mask)[0] if return_indices else None
    
    return {
        'false_positives': {
            'count': fp_mask.sum(),
            'indices': fp_indices,
            'percentage': fp_mask.sum() / len(y_true) * 100
        },
        'false_negatives': {
            'count': fn_mask.sum(),
            'indices': fn_indices,
            'percentage': fn_mask.sum() / len(y_true) * 100
        },
        'total_errors': (fp_mask | fn_mask).sum(),
        'error_rate': (fp_mask | fn_mask).sum() / len(y_true) * 100
    }


def visualize_error_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: pd.DataFrame,
    y_proba: Optional[np.ndarray] = None,
    max_samples: int = 10,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Visualize or print samples where the model made mistakes.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    X : pd.DataFrame
        Feature dataframe
    y_proba : np.ndarray, optional
        Predicted probabilities
    max_samples : int
        Maximum number of error samples to display
    save_path : str, optional
        Path to save the error samples as CSV
    
    Returns:
    --------
    pd.DataFrame : Error samples dataframe
    """
    errors = identify_errors(y_true, y_pred, X)
    
    # Combine FP and FN
    fp_indices = errors['false_positives']['indices']
    fn_indices = errors['false_negatives']['indices']
    
    all_error_indices = np.concatenate([fp_indices, fn_indices])[:max_samples]
    
    if len(all_error_indices) == 0:
        print("[INFO] No errors found!")
        return pd.DataFrame()
    
    # Create error dataframe
    error_df = X.iloc[all_error_indices].copy()
    error_df['true_label'] = y_true[all_error_indices]
    error_df['predicted_label'] = y_pred[all_error_indices]
    error_df['error_type'] = ['FP' if idx in fp_indices else 'FN' for idx in all_error_indices]
    
    if y_proba is not None:
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            error_df['predicted_probability'] = y_proba[all_error_indices, 1]
        else:
            error_df['predicted_probability'] = y_proba[all_error_indices]
    
    # Display
    print("\n" + "="*60)
    print(f"ERROR SAMPLES (showing {len(all_error_indices)} of {errors['total_errors']} total errors)")
    print("="*60)
    print(error_df.to_string())
    print("="*60 + "\n")
    
    if save_path:
        error_df.to_csv(save_path, index=False)
        print(f"[INFO] Error samples saved to {save_path}")
    
    return error_df


def analyze_error_patterns(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: pd.DataFrame,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Analyze common patterns in errors.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    X : pd.DataFrame
        Feature dataframe
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the figure
    """
    errors = identify_errors(y_true, y_pred, X)
    
    fp_indices = errors['false_positives']['indices']
    fn_indices = errors['false_negatives']['indices']
    
    # Create error labels
    error_labels = np.zeros(len(y_true))
    error_labels[fp_indices] = 1  # FP
    error_labels[fn_indices] = 2   # FN
    
    # Get numerical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    n_cols = min(len(numerical_cols), 6)
    
    if n_cols == 0:
        print("[WARN] No numerical features for error pattern analysis.")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols[:6]):
        if idx >= len(axes):
            break
        
        # Create data for plotting
        plot_data = pd.DataFrame({
            col: X[col],
            'Error Type': ['Correct' if label == 0 else ('FP' if label == 1 else 'FN') 
                          for label in error_labels]
        })
        
        # Boxplot
        sns.boxplot(data=plot_data, x='Error Type', y=col, ax=axes[idx], palette='Set2')
        axes[idx].set_title(f'{col} Distribution by Error Type', fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Error Pattern Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("ERROR PATTERN SUMMARY")
    print("="*60)
    print(f"False Positives: {errors['false_positives']['count']} ({errors['false_positives']['percentage']:.2f}%)")
    print(f"False Negatives: {errors['false_negatives']['count']} ({errors['false_negatives']['percentage']:.2f}%)")
    print(f"Total Errors: {errors['total_errors']} ({errors['error_rate']:.2f}%)")
    print("="*60 + "\n")


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def run_full_visualization_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    models: Dict[str, Any],
    models_results: Optional[Dict[str, Dict[str, float]]] = None,
    best_model_name: Optional[str] = None,
    save_dir: Optional[str] = None
) -> None:
    """
    Run the complete visualization pipeline.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    y_train : pd.Series
        Training labels
    y_test : pd.Series
        Test labels
    models : dict
        Dictionary of trained models {model_name: model_object}
    models_results : dict, optional
        Dictionary with model metrics
    best_model_name : str, optional
        Name of the best model (if None, uses first model)
    save_dir : str, optional
        Directory to save all plots
    """
    import os
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Determine best model
    if best_model_name is None:
        best_model_name = list(models.keys())[0]
    
    best_model = models[best_model_name]
    
    print("="*60)
    print("STARTING VISUALIZATION PIPELINE")
    print("="*60)
    
    # 1. Data Understanding
    print("\n[1/5] Data Understanding...")
    plot_target_distribution(
        pd.concat([y_train, y_test]),
        save_path=os.path.join(save_dir, 'target_distribution.png') if save_dir else None
    )
    
    # plot_feature_distributions removed per user request
    # plot_feature_distributions(
    #     X_train,
    #     save_path=os.path.join(save_dir, 'feature_distributions.png') if save_dir else None
    # )
    
    plot_correlation_heatmap(
        X_train,
        y_train,
        save_path=os.path.join(save_dir, 'correlation_heatmap.png') if save_dir else None
    )
    
    # 2. Model Evaluation
    print("\n[2/5] Model Evaluation...")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    
    plot_model_evaluation(
        y_test.values,
        y_pred,
        y_proba,
        model_name=best_model_name,
        save_path=os.path.join(save_dir, f'{best_model_name}_evaluation.png') if save_dir else None
    )
    
    # 3. Model Comparison
    if models_results:
        print("\n[3/5] Model Comparison...")
        create_model_comparison_table(
            models_results,
            save_path=os.path.join(save_dir, 'model_comparison.csv') if save_dir else None
        )
        plot_model_comparison(
            models_results,
            save_path=os.path.join(save_dir, 'model_comparison.png') if save_dir else None
        )
    else:
        print("\n[3/5] Model Comparison skipped (no results provided)")
    
    # 4. Model Explainability
    print("\n[4/5] Model Explainability...")
    plot_feature_importance(
        best_model,
        X_test.columns.tolist(),
        save_path=os.path.join(save_dir, 'feature_importance.png') if save_dir else None
    )
    
    # SHAP (optional)
    if SHAP_AVAILABLE:
        plot_shap_summary(
            best_model,
            X_test.sample(min(100, len(X_test))),
            save_path=os.path.join(save_dir, 'shap_summary.png') if save_dir else None
        )
    
    # 5. Error Analysis - removed per user request
    # print("\n[5/5] Error Analysis...")
    # visualize_error_samples(
    #     y_test.values,
    #     y_pred,
    #     X_test,
    #     y_proba,
    #     save_path=os.path.join(save_dir, 'error_samples.csv') if save_dir else None
    # )
    # 
    # analyze_error_patterns(
    #     y_test.values,
    #     y_pred,
    #     X_test,
    #     save_path=os.path.join(save_dir, 'error_patterns.png') if save_dir else None
    # )
    
    print("\n" + "="*60)
    print("VISUALIZATION PIPELINE COMPLETE")
    print("="*60)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_model_metrics(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Extract metrics from a model.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    
    Returns:
    --------
    dict : Dictionary with metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Handle binary classification probabilities
    if y_proba.ndim > 1 and y_proba.shape[1] > 1:
        y_proba_binary = y_proba[:, 1]
    else:
        y_proba_binary = y_proba
    
    # Calculate ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba_binary)
    roc_auc = auc(fpr, tpr)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'roc_auc': roc_auc,
        'pr_auc': average_precision_score(y_test, y_proba_binary)
    }

