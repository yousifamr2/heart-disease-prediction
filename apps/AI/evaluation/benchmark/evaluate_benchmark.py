import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup paths
EVAL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL_ROOT))

from config import BENCHMARK_DIR

def evaluate_benchmark():
    print("="*50)
    print("Starting Benchmark Evaluation")
    print("="*50)
    
    json_path = BENCHMARK_DIR / 'benchmark_results.json'
    if not json_path.exists():
        print("No benchmark_results.json found. Skipping benchmark comparison.")
        sys.exit(0)
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    if "benchmark" not in data:
        print("Invalid benchmark_results.json format.")
        sys.exit(1)
        
    records = []
    for model_name, metrics in data["benchmark"].items():
        metrics["Model"] = model_name
        records.append(metrics)
        
    df = pd.DataFrame(records)
    df.to_csv(BENCHMARK_DIR / 'benchmark_comparison.csv', index=False)
    
    # Melt for plotting
    df_melt = df.melt(id_vars=["Model"], var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melt, x="Metric", y="Score", hue="Model", palette="Set2")
    plt.title("Benchmark Model Comparison")
    plt.ylabel("Score (%)")
    plt.ylim(0, 100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(BENCHMARK_DIR / 'benchmark_comparison.png', bbox_inches='tight')
    plt.close()
    
    print("Benchmark Evaluation Complete.")

if __name__ == "__main__":
    evaluate_benchmark()
