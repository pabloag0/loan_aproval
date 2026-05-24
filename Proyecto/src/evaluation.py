from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import os
from src import preprocess as pp


def get_confusion_dir():
    project_dir = os.path.dirname(os.path.dirname(__file__))
    save_dir = os.path.join(project_dir, "results", "confusion_matrix")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def safe_filename(name):
    return "".join(
        char.lower() if char.isalnum() else "_"
        for char in name
    ).strip("_")


def evaluate(y_pred, y_test, mostrar=False, show_matrix=False, matrix_name=None):

    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred) * 100
    recall = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100

    if mostrar:
        print(f"Accuracy: {accuracy}%")
        print(f"Precision: {precision}%")
        print(f"Recall: {recall}%")
        print(f"F1: {f1}%")

    #Confusion matrix (label)
    if show_matrix:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        if matrix_name is not None:
            plt.title(matrix_name)
            plt.savefig(
                os.path.join(get_confusion_dir(), f"{safe_filename(matrix_name)}.png"),
                dpi=150,
                bbox_inches="tight"
            )
        else:
            plt.title("Confusion matrix")
        plt.show(block=False)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
