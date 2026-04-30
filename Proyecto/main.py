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

def main():
    os.system('clear')

    print('Cargando datos...')
    df = pd.read_csv(directorio + "data/loan_data.csv")

    X_train, X_test, y_train, y_test = pp.preprocess(df, split=True, lr=True)

    w_in = np.zeros(X_train.shape[1])
    w, b, j = lr.train(X_train, y_train, w_in, lambda_=0.01, reg=True, num_iters=10000)
    w2, b2, j2 = lr.train(X_train, y_train, w_in, lambda_=0.01, reg=False, num_iters=10000)

    y_pred = lr.predict(X_test, w, b)
    y_pred2 = lr.predict(X_test, w2, b2)
    accuracy = np.mean(y_test== y_pred)*100
    accuracy2 = np.mean(y_test== y_pred2)*100
    print(f'Accuracy (Regularized): {accuracy}')
    print(f'Accuracy (Non-Regularized): {accuracy2}')

    input('Pulsa Enter para cerrar el programa..')

main()