import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from src import preprocess as pp



def evaluate(y_pred, y_test, show_matrix=True):

    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred) * 100
    recall = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100

    print(f"Acuraccy: {accuracy}%")
    print(f"Precision: {precision}%")
    print(f"Recall: {recall}%")
    print(f"F1: {f1}%")

    #Confusion matrix (label)
    if show_matrix:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title("Confusion matrix")
        plt.show(block=False)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_defaults(X_test, y_pred):
    #Confusion matrix (defaults)
    defaults = (X_test["previous_loan_defaults_on_file_Yes"] > 0).astype(int)
    cm = confusion_matrix(defaults, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap='Blues')
    plt.title("Defaults vs prediction")
    plt.xlabel("Predicción loan_status")
    plt.ylabel("Default previo")
    plt.show()
