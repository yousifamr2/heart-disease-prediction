import pandas as pd
from Analytics import load_data
from sklearn.preprocessing import StandardScaler

def find_outliers(data, column):
    if not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"{column} must be numeric")

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = data[(data[column] < lower) | (data[column] > upper)]
    inliers = data[(data[column] >= lower) & (data[column] <= upper)]

    stats = {
        "column": column,
        "lower_bound": lower,
        "upper_bound": upper,
        "inliers": inliers.shape[0],
        "outliers": outliers.shape[0]
    }

    return outliers, inliers, stats



if __name__ == "__main__":
    train_df = load_data("data/heart_statlog_cleveland_hungary_final.csv")
    outliers, inliers, stats = find_outliers(train_df, "age")
    print(stats)
