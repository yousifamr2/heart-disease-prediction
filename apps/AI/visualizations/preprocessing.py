import pandas as pd
import numpy as np
from Analytics import load_data
from sklearn.preprocessing import StandardScaler

def find_outliers(data, column):
    
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

    return stats

def handdling_outliers(train_df, first_column = "cholesterol", second_column ="resting bp s", third_column ="oldpeak"):
    
    train_df.loc[train_df[first_column] == 0, first_column] = np.nan
    train_df[first_column].fillna(train_df[first_column].median(), inplace=True)

    train_df[second_column] = train_df[second_column].clip(90, 170)
    
    train_df.loc[train_df[third_column] < 0, third_column] = 0

    return train_df


if __name__ == "__main__":
    train_df = load_data("data/heart_statlog_cleveland_hungary_final.csv")
    outliers, inliers, stats = find_outliers(train_df, "age")
    print(stats)
