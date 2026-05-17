import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from src import preprocess as pp



def evaluate(X_test, y_pred, y_test):

    acc = np.mean(y_pred == y_test)*100

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


# Lo he comentado porque no se que hace aqui, si falla descomenta
"""
def get_approved_loans_preprocessed(df: pd.DataFrame, approved_label=1):
    ""Filtra los creditos aprobados, los preprocesa y devuelve X e y.""
    if 'loan_status' not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'loan_status'.")

    # Codificamos primero el dataset completo para conservar exactamente
    # las mismas columnas que se usaron al entrenar.
    encoded_df = pp.encode_categoricals(df.copy())
    approved_df = encoded_df[encoded_df['loan_status'] == approved_label].copy()

    if approved_df.empty:
        raise ValueError('No hay filas con loan_status igual al valor indicado.')

    X = approved_df.drop(columns=['loan_status']).to_numpy()
    y = approved_df['loan_status'].to_numpy().reshape(-1, 1)

    return X, y

def test(y_test, y_pred):
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred) 
    ConfusionMatrixDisplay(cm).plot()
    plt.title('Matriz de confusión')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
"""
