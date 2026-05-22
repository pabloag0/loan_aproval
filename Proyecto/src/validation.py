import pandas as pd
import numpy as np
from src import evaluation as ev
from sklearn.model_selection import StratifiedKFold

##############################################################################
# CROSS VALIDATION

def cross_validate(X, y, train, folds=5):
    """Valida un modelo con Stratified K-Fold.

    La funcion train debe recibir X_train, X_val, y_train y devolver y_pred
    para X_val. Asi se puede usar con modelos propios, no solo con sklearn.
    """

    y_flat = np.asarray(y).reshape(-1)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_flat), start=1):
        if hasattr(X, "iloc"):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
        else:
            X_array = np.asarray(X)
            X_train = X_array[train_idx]
            X_val = X_array[val_idx]

        if hasattr(y, "iloc"):
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]
        else:
            y_array = np.asarray(y)
            y_train = y_array[train_idx]
            y_val = y_array[val_idx]

        y_val = np.asarray(y_val).reshape(-1)
        y_pred = np.asarray(train(X_train, X_val, y_train)).reshape(-1)

        print(f"Fold {fold}")
        metrics = ev.evaluate(y_pred, y_val)
        fold_metrics.append(metrics)

    summary = {}
    for metric_name in fold_metrics[0]:
        values = np.array([metrics[metric_name] for metrics in fold_metrics])
        summary[metric_name] = {
            "mean": values.mean(),
            "std": values.std(),
        }

    print("=" * 50)
    print("RESUMEN CROSS VALIDATION")
    print("=" * 50)
    for metric_name, values in summary.items():
        print(
            f"{metric_name}: "
            f"{values['mean']:.2f}% +/- {values['std']:.2f}%"
        )

    return summary

# Dijo en clase que sería interesante implementarlo (mas nota)
def gridsearch():
    pass
