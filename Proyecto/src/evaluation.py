import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from src import preprocess as pp



def evaluate(X_test, y_pred, y_test):

    # accuracy
    acc = np.mean(y_pred == y_test)*100

    # precision

    # recall

    # f1


    print(f"Acuraccy: {acc}%")

    #Confusion matrix (label)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion matrix")
    plt.show()

    #Confusion matrix (defaults)
    defaults = (X_test["previous_loan_defaults_on_file_Yes"] > 0).astype(int)
    cm = confusion_matrix(defaults, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap='Blues')
    plt.title("Defaults vs prediction")
    plt.xlabel("Predicción loan_status")
    plt.ylabel("Default previo")
    plt.show()