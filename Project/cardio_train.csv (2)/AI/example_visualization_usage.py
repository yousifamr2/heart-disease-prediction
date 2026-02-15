"""
Example Usage of Visualization Module
=====================================
This script demonstrates how to use the visualization module with your existing ML pipeline.
"""

from Analytics import load_data
from preprocessing import handdling_outliers
from model import Model
from visualization import (
    run_full_visualization_pipeline,
    extract_model_metrics,
    create_formatted_results_table
)
import pandas as pd
import numpy as np
import time


def main():
    """Main function demonstrating visualization usage."""
    
    # Load and prepare data
    print("Loading data...")
    train_df = load_data("data/heart_statlog_cleveland_hungary_final.csv")
    train_df = handdling_outliers(train_df)
    
    # Initialize model
    model = Model(
        target_col="target",
        random_state=42,
        test_size=0.2
    )
    
    # Prepare data
    X_train, X_test, y_train, y_test = model.prepare_data(train_df)
    
    # Train models
    print("\nTraining models...")
    model.training_model(X_train, y_train, X_test, y_test)
    
    # Extract metrics for all models
    print("\nExtracting model metrics...")
    models_results = {}
    training_times = {}
    
    for model_name, trained_model in model.best_models.items():
        start_time = time.time()
        metrics = extract_model_metrics(trained_model, X_test, y_test)
        training_times[model_name] = time.time() - start_time
        models_results[model_name] = metrics
    
    # Find best model
    best_model_name = max(models_results.keys(), 
                         key=lambda x: models_results[x]['accuracy'])
    print(f"\nBest model: {best_model_name} (Accuracy: {models_results[best_model_name]['accuracy']:.4f})")
    
    # Save best model
    print("\n" + "="*60)
    print("Saving Best Model")
    print("="*60)
    best_name, best_metrics = model.save_best_model(
        X_test=X_test,
        y_test=y_test,
        save_path="models/best_model.pkl",
        metric="accuracy"
    )

    print("\n" + "="*60)
    print("Running Full Visualization Pipeline")
    print("="*60)
    
    run_full_visualization_pipeline(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        models=model.best_models,
        models_results=models_results,
        best_model_name=best_model_name,
        save_dir="visualizations"  # All plots will be saved here
    )
    
    # Generate formatted results table (CSV output)
    print("\nGenerating formatted results table...")
    formatted_table = create_formatted_results_table(
        models_results,
        metrics=['accuracy', 'precision', 'recall', 'f1_score'],
        primary_metric='accuracy',
        save_path='visualizations/formatted_results_table.csv'
    )
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

