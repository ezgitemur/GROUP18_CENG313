"""Model definitions and cross-validation utilities for the CMI BFRB detection project."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from src.utils import competition_score


def make_random_forest():
    """Random forest baseline; missing values are filled with the column median."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
    )


def cross_validate(model_fn, X, y, groups, bfrb_gestures, n_splits=5):
    """Evaluate a model with subject-based GroupKFold.
    Returns the fold scores and the out-of-fold predictions."""
    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = np.empty(len(y), dtype=object)
    scores = []

    for fold, (train_idx, valid_idx) in enumerate(gkf.split(X, y, groups)):
        model = model_fn()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[valid_idx])
        oof_pred[valid_idx] = pred

        score = competition_score(y.iloc[valid_idx], pred, bfrb_gestures)
        scores.append(score)
        print(f"Fold {fold + 1}: {score:.4f}")

    print(f"Mean: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores, oof_pred
