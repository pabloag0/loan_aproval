# debe ser el programa principal que al ejecutarse con python main.py reproduzca todos los resultados:
# carga de datos, preprocesado, entrenamiento de todos los modelos, evaluación y guardado de gráficas. 
# El enunciado lo exige explícitamente.

# smote

import random
import numpy as np
import pandas as pd
from src.models import neural_network as nn
from src.models import logistic_regression as lr
from src.models import deep_neural_network as dnn
from src import preprocess as pp
from src import eda as eda
from src import evaluation as ev
import matplotlib.pyplot as plt
import os
import time

#directorio = '/Users/pabloag/uni/loan_aproval/Proyecto/'
directorio = 'D:/onedrive/OneDrive - Universidad Complutense de Madrid (UCM)/uni/3/2/AA/proyecto_git/loan_aproval/Proyecto/'


def logistic_regression(X_train, X_test, y_train):

    w, b, j = lr.train(X_train, y_train, np.zeros(X_train.shape[1]), 0, alpha=0.1, num_iters=300, lambda_=1)
    y_pred = lr.predict(X_test, w, b)

    return y_pred

def neural_network(X_train, X_test, y_train):
    
    theta1, theta2 = nn.train(X_train, y_train, num_labels=1, alpha=0.1, num_iters=300, hidden_size=16 , reg=0, input_size=X_train.shape[1])
    y_pred = nn.predict(theta1, theta2, X_test)

    return y_pred

def deep_neural_network(X_train, X_test, y_train, y_test):
    
    dnn.ejecutar(X_train, X_test, y_train, y_test)
    
    pass

def main():
    os.system('cls')

    print('Cargando datos...')
    df = pd.read_csv(directorio + "data/loan_data.csv")

    print('Preprocesando datos...')
    X_train, X_test, y_train, y_test = pp.preprocess(df, split=True, lr=True)

    print('Entrenando regresión logística: ')
    #y_pred = logistic_regression(X_train, X_test, y_train)

    #print('Entrenando red neuronal...')
    #y_pred = neural_network(X_train, X_test, y_train)

    #acc = np.mean(y_pred == y_test)*100
    #print(f"Logistic Regression Accuracy: {acc:.2f}%")

    deep_neural_network(X_train, X_test, y_train, y_test)

    input('Pulsa Enter para cerrar el programa..')

main()
