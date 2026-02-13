from Analytics import load_data
from preprocessing import find_outliers
from preprocessing import handdling_outliers
from Analytics import load_data
import matplotlib.pyplot as plt
import seaborn as sns
from visualization import (
    run_full_visualization_pipeline,
    extract_model_metrics,
    create_formatted_results_table,
    plot_correlation_heatmap
)
def visuailation_outliers(data, column):

    plt.figure(figsize=(6,3))
    sns.boxplot(x=data[column], data= data)
    plt.title(f"Boxplot of {column}")
    plt.show()

    plt.figure(figsize=(5.23,3))
    sns.histplot(x=data[column], data= data)
    plt.title(f"Boxplot of {column}")
    plt.show()
    
    print("="*50)

if __name__ == "__main__":
    print(plot_correlation_heatmap.__doc__)
#     train_df = load_data("data/heart_statlog_cleveland_hungary_final.csv")
#     train_df = handdling_outliers(train_df)
#     for col in train_df.columns:
#         stats = find_outliers(train_df, col)
#         print(stats)
#         visuailation_outliers(train_df, col)