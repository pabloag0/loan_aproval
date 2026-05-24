import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src import preprocess as pp
from src.models import neural_network as nn


def get_curves_dir():
    project_dir = os.path.dirname(os.path.dirname(__file__))
    save_dir = os.path.join(project_dir, "results", "figures", "curves")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def safe_filename(title):
    return (
        title.lower()
        .replace(" - ", "_")
        .replace(" ", "_")
        .replace(":", "")
    )


def plot_training_curve(J_history, title):
    plt.figure()
    plt.plot(J_history)
    plt.xlabel("Iteracion")
    plt.ylabel("Coste")
    plt.title(title)
    plt.grid(True)
    plt.savefig(
        os.path.join(get_curves_dir(), f"{safe_filename(title)}.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.show()


def get_rows(data, indexes):
    if hasattr(data, "iloc"):
        return data.iloc[indexes]
    return np.asarray(data)[indexes]


def score_f1(y_true, y_pred):
    return f1_score(
        np.asarray(y_true).reshape(-1),
        np.asarray(y_pred).reshape(-1),
        zero_division=0
    ) * 100


def plot_learning_curve(x_values, train_scores, val_scores, title, xlabel):
    plt.figure()
    plt.plot(x_values, train_scores, marker="o", label="Train")
    plt.plot(x_values, val_scores, marker="o", label="Validacion")
    plt.xlabel(xlabel)
    plt.ylabel("F1")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(get_curves_dir(), f"{safe_filename(title)}.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.show()


def learning_curve_cv(X, y, train, train_sizes=None, folds=3, lr=False, undersampling=False, title="Learning curve"):
    if train_sizes is None:
        train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0]

    y_flat = np.asarray(y).reshape(-1)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    train_scores = []
    val_scores = []

    for size in train_sizes:
        train_fold_scores = []
        val_fold_scores = []

        for train_idx, val_idx in cv.split(X, y_flat):
            X_train = get_rows(X, train_idx)
            X_val = get_rows(X, val_idx)
            y_train = get_rows(y, train_idx)
            y_val = get_rows(y, val_idx)

            if size < 1.0:
                X_train, _, y_train, _ = train_test_split(
                    X_train,
                    y_train,
                    train_size=size,
                    random_state=42,
                    stratify=y_train
                )

            X_train, X_val, y_train, y_val = pp.preprocess(
                X_train,
                X_val,
                y_train,
                y_val,
                lr=lr,
                undersampling=undersampling
            )

            y_pred_train = train(X_train, X_train, y_train)
            y_pred_val = train(X_train, X_val, y_train)

            train_fold_scores.append(score_f1(y_train, y_pred_train))
            val_fold_scores.append(score_f1(y_val, y_pred_val))

        train_scores.append(np.mean(train_fold_scores))
        val_scores.append(np.mean(val_fold_scores))

    plot_learning_curve(
        train_sizes,
        train_scores,
        val_scores,
        title,
        "Proporcion del conjunto de entrenamiento"
    )
    return train_scores, val_scores


def validation_curve_cv(
    X,
    y,
    hidden_sizes=None,
    folds=3,
    undersampling=False,
    alpha=0.1,
    num_iters=1200,
    reg=0
):
    if hidden_sizes is None:
        hidden_sizes = [8, 16, 32]

    y_flat = np.asarray(y).reshape(-1)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    train_scores = []
    val_scores = []

    for hidden_size in hidden_sizes:
        train_fold_scores = []
        val_fold_scores = []

        for train_idx, val_idx in cv.split(X, y_flat):
            X_train = get_rows(X, train_idx)
            X_val = get_rows(X, val_idx)
            y_train = get_rows(y, train_idx)
            y_val = get_rows(y, val_idx)

            X_train, X_val, y_train, y_val = pp.preprocess(
                X_train,
                X_val,
                y_train,
                y_val,
                undersampling=undersampling
            )

            theta1, theta2, _ = nn.train(
                X_train,
                y_train,
                num_labels=1,
                alpha=alpha,
                num_iters=num_iters,
                hidden_size=hidden_size,
                reg=reg,
                input_size=X_train.shape[1]
            )

            y_pred_train = nn.predict(theta1, theta2, X_train)
            y_pred_val = nn.predict(theta1, theta2, X_val)

            train_fold_scores.append(score_f1(y_train, y_pred_train))
            val_fold_scores.append(score_f1(y_val, y_pred_val))

        train_scores.append(np.mean(train_fold_scores))
        val_scores.append(np.mean(val_fold_scores))

    plot_learning_curve(
        hidden_sizes,
        train_scores,
        val_scores,
        "Validation curve - Red neuronal",
        "Neuronas en capa oculta"
    )

    return train_scores, val_scores
