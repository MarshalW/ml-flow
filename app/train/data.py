# data.py
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split


def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    df["think"] = df.get("think", "").fillna("").astype(str)
    dataset = [
        {
            "instruction": row["prompt"],
            "input": "",
            "output": row["response"],
            "think": row["think"],
        }
        for _, row in df.iterrows()
    ]
    train_df, eval_df = train_test_split(
        pd.DataFrame(dataset), test_size=0.1, random_state=42
    )
    return Dataset.from_pandas(train_df), Dataset.from_pandas(eval_df)
