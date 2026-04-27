import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from src import preprocess as pp


def get_approved_loans_preprocessed(df: pd.DataFrame, approved_label=1):
    """Filtra los creditos aprobados, los preprocesa y devuelve X e y."""
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


##############################################################################
# CROSS VALIDATION

def cross_validate(X, y):




    pass

#############################################
# ONLY DEBUGGING PURPOSES, DELETE THIS LATER#
#############################################
import sys                                  #
sys.dont_write_bytecode = True              #
#############################################