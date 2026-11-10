"""Evaluation and helper utilities for the CMI BFRB detection project."""

import numpy as np
from sklearn.metrics import f1_score


def competition_score(y_true, y_pred, bfrb_gestures):
    """Competition metric: average of binary F1 (BFRB vs non-BFRB) and macro F1
    over BFRB gestures, with all non-BFRB gestures merged into one class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_is_bfrb = np.isin(y_true, bfrb_gestures)
    pred_is_bfrb = np.isin(y_pred, bfrb_gestures)
    binary_f1 = f1_score(true_is_bfrb, pred_is_bfrb)

    true_merged = np.where(true_is_bfrb, y_true, "non_target")
    pred_merged = np.where(pred_is_bfrb, y_pred, "non_target")
    macro_f1 = f1_score(true_merged, pred_merged, average="macro")

    return (binary_f1 + macro_f1) / 2
