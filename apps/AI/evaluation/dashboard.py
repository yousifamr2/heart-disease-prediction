import sys
import os
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image

# Setup paths
EVAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_ROOT))

from config import (
    FIGURES_ECG_DIR, FIGURES_LLM_DIR, SCORECARDS_DIR, BENCHMARK_DIR
)

st.set_page_config(page_title="Heart Disease Prediction Evaluation", layout="wide")
st.title("Heart Disease Prediction System - Evaluation Dashboard")

tab1, tab2, tab3 = st.tabs(["ECG Dashboard", "LLM Dashboard", "Benchmark Dashboard"])

def display_png(path, caption=""):
    if path.exists():
        img = Image.open(path)
        st.image(img, caption=caption, use_column_width=True)
    else:
        st.warning(f"Image not found: {path.name}. Run the evaluation scripts first.")

with tab1:
    st.header("ECG Model Evaluation")
    
    scorecard_path = SCORECARDS_DIR / 'ecg_scorecard.csv'
    if scorecard_path.exists():
        df_score = pd.read_csv(scorecard_path)
        st.subheader("ECG Scorecard")
        
        # Display as metric cards
        cols = st.columns(len(df_score))
        for i, row in df_score.iterrows():
            cols[i].metric(label=row["Metric"], value=f"{row['Score']:.2f}")
            
        # Plotly Bar Chart
        fig = px.bar(df_score, x="Metric", y="Score", title="ECG Metrics Overview", text="Score", color="Score", color_continuous_scale="Viridis")
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("ECG Scorecard not found. Please run evaluate_ecg.py first.")
        
    st.subheader("Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        display_png(FIGURES_ECG_DIR / 'confusion_matrix.png', "Confusion Matrix (Top 10)")
        display_png(FIGURES_ECG_DIR / 'roc_curves.png', "ROC Curves (Top 10)")
        display_png(FIGURES_ECG_DIR / 'confidence_histogram.png', "Prediction Confidence Distribution")
        display_png(FIGURES_ECG_DIR / 'threshold_vs_f1.png', "Threshold Optimization (F1)")
        
    with col2:
        display_png(FIGURES_ECG_DIR / 'class_distribution.png', "Class Distribution")
        display_png(FIGURES_ECG_DIR / 'precision_recall_curves.png', "Precision-Recall Curves (Top 10)")
        display_png(FIGURES_ECG_DIR / 'reliability_diagram.png', "Reliability Diagram")
        display_png(FIGURES_ECG_DIR / 'threshold_vs_recall.png', "Threshold Optimization (Recall)")

with tab2:
    st.header("LLM Evaluation")
    
    scorecard_path = SCORECARDS_DIR / 'llm_scorecard.csv'
    if scorecard_path.exists():
        df_score = pd.read_csv(scorecard_path)
        st.subheader("LLM Scorecard")
        
        cols = st.columns(len(df_score))
        for i, row in df_score.iterrows():
            cols[i].metric(label=row["Metric"], value=f"{row['Score']:.2f}")
            
        fig = px.bar(df_score, x="Metric", y="Score", title="LLM Metrics Overview", text="Score", color="Score", color_continuous_scale="Blues")
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("LLM Scorecard not found. Please run evaluate_llm.py first.")
        
    st.subheader("Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        display_png(FIGURES_LLM_DIR / 'latency_distribution.png', "Latency Distribution")
        display_png(FIGURES_LLM_DIR / 'readability_distribution.png', "Readability Distribution (Flesch)")
        
    with col2:
        display_png(FIGURES_LLM_DIR / 'grounding_distribution.png', "Clinical Grounding Distribution")
        display_png(FIGURES_LLM_DIR / 'output_length_distribution.png', "Output Length Distribution")

with tab3:
    st.header("Benchmark Comparison")
    
    bench_csv = BENCHMARK_DIR / 'benchmark_comparison.csv'
    if bench_csv.exists():
        df_bench = pd.read_csv(bench_csv)
        st.dataframe(df_bench, use_container_width=True)
        
        df_melt = df_bench.melt(id_vars=["Model"], var_name="Metric", value_name="Score")
        fig = px.bar(df_melt, x="Metric", y="Score", color="Model", barmode="group", title="Benchmark Models Comparison")
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
        
        display_png(BENCHMARK_DIR / 'benchmark_comparison.png', "Static Benchmark Chart")
    else:
        st.warning("Benchmark CSV not found. Please run evaluate_benchmark.py first.")
