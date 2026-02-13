import numpy as np
import pandas as pd


def load_data(path):
    return pd.read_csv(path, sep=",")

# def preprocess_data(data):
#     data = data.drop(columns=["id"])
#     return data 


def save_data(data, path):
    data.to_csv(path, sep=";")

def some_stats(data, function):
    funcs = {
        "describe": data.describe,
        "info": lambda: data.info(),
        "head": data.head,
        "tail": data.tail,
        "nunique": data.nunique,
        "unique": lambda: {col: data[col].unique() for col in data.columns}
    }
    if function in funcs:
        return funcs[function]()
    else:
        return "Invalid function"


if __name__ == "__main__":
    train_df = load_data("heart_statlog_cleveland_hungary_final.csv")
    save_data(train_df, "heart_statlog_cleveland_hungary_final_processed.csv")


    print(some_stats(train_df, "describe"))
    print(some_stats(train_df, "info"))
    print(some_stats(train_df, "head"))
    print(some_stats(train_df, "tail"))


