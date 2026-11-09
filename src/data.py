"""Data loading and preprocessing utilities for the CMI BFRB detection project."""

import pandas as pd


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
