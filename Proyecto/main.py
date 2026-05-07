# debe ser el programa principal que al ejecutarse con python main.py reproduzca todos los resultados:
# carga de datos, preprocesado, entrenamiento de todos los modelos, evaluación y guardado de gráficas. 
# El enunciado lo exige explícitamente.

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

directorio = '/Users/pabloag/uni/loan_aproval/Proyecto/'

def logistic_regression():
    pass

def neural_network():
    pass

def deep_neural_network():
    pass

def main():
    os.system('clear')

    print('Cargando datos...')
    df = pd.read_csv(directorio + "data/loan_data.csv")

    print('Preprocesando datos...')
    X_train, X_test, y_train, y_test = pp.preprocess(df, split=True, lr=True, des=False)


    
    w, b, j = lr.train(X_train, y_train, np.zeros(X_train.shape[1]), 0, alpha=0.01, num_iters=1000, lambda_=1)
    y_pred = lr.predict(X_test, w, b)
    acc = np.mean(y_pred == y_test)*100
    print(f"Logistic Regression Accuracy: {acc:.2f}%")

    input('Pulsa Enter para cerrar el programa..')

main()
