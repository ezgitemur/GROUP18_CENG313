"""Data loading and preprocessing utilities for the CMI BFRB detection project."""

import pandas as pd
import numpy as np


def load_train(data_dir="data"):
    """Load the training sensor data and participant demographics."""
    train = pd.read_csv(f"{data_dir}/train.csv")
    demo = pd.read_csv(f"{data_dir}/train_demographics.csv")
    return train, demo


def get_sensor_groups(df):
    """Return column names grouped by sensor type."""
    return {
        "IMU": [c for c in df.columns if c.startswith(("acc_", "rot_"))],
        "Thermopile": [c for c in df.columns if c.startswith("thm_")],
        "ToF": [c for c in df.columns if c.startswith("tof_")],
    }

def get_sequence_labels(df):
    """Return one row per sequence with its gesture, sequence type and subject."""
    return df.groupby("sequence_id")[["gesture", "sequence_type", "subject"]].first()


def extract_features(df, feature_cols):
    """Summarize each sequence into one row using statistics of the given columns."""
    agg = df.groupby("sequence_id")[feature_cols].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    return agg

def summarize_tof(df):
    """Summarize each ToF sensor's 64 pixels at every time step into two values:
    mean distance of detected pixels and fraction of pixels with no detection (-1)."""
    summary = df[["sequence_id"]].copy()
    for i in range(1, 6):
        cols = [f"tof_{i}_v{j}" for j in range(64)]
        values = df[cols]
        summary[f"tof_{i}_mean"] = values.replace(-1, np.nan).mean(axis=1)
        summary[f"tof_{i}_nodetect"] = (values == -1).sum(axis=1) / values.notna().sum(axis=1)
    return summary
