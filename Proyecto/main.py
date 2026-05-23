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
from src import validation as val
import matplotlib.pyplot as plt
import os
import time

directorio = '/Users/pabloag/uni/loan_aproval/Proyecto/'
#directorio = 'D:/onedrive/OneDrive - Universidad Complutense de Madrid (UCM)/uni/3/2/AA/proyecto_git/loan_aproval/Proyecto/'


def logistic_regression(X_train, X_val, y_train):
    w, b, _ = lr.train(
        X_train,
        y_train,
        np.zeros(X_train.shape[1]),
        0,
        alpha=0.1,
        num_iters=3000,
        lambda_=1
    )

    y_pred = lr.predict(X_val, w, b)

    return y_pred


def neural_network(X_train, X_val, y_train):
    
    theta1, theta2 = nn.train(
        X_train, 
        y_train, 
        num_labels=1, 
        alpha=0.1, 
        num_iters=3000, 
        hidden_size=16, 
        reg=0, 
        input_size=X_train.shape[1])

    y_pred = nn.predict(theta1, theta2, X_val)

    return y_pred


def deep_neural_network(X_train, X_val, y_train):
    
    y_pred = dnn.ejecutar(X_train, X_val, y_train)

    return y_pred


def main():
    os.system('clear')
    from src import validation as val

    print('Cargando datos...')
    df = pd.read_csv(directorio + "data/loan_data.csv")

    X_train, X_test, y_train, y_test = pp.split(df)

    print("Regresión logística:")
    val.cross_validate(X_train, y_train, logistic_regression, folds=5, lr=True)
    
    print("Red neuronal:")
    val.cross_validate(X_train, y_train, neural_network, folds=5)
    
    print("Red neuronal profunda:")
    val.cross_validate(X_train, y_train, deep_neural_network, folds=5)

    X_train_lr, X_test_lr, y_train_lr, y_test_lr = pp.preprocess(X_train, X_test, y_train, y_test, lr=True)
    X_train, X_test, y_train, y_test = pp.preprocess(X_train, X_test, y_train, y_test)

    # refit
    y_pred_lr = logistic_regression(X_train_lr, X_test_lr, y_train_lr)
    y_pred_nn = neural_network(X_train, X_test, y_train)

    print("Regresión logística en test:")
    ev.evaluate(y_pred_lr, y_test, mostrar=True)
    print("Red neuronal en test:")
    ev.evaluate(y_pred_nn, y_test, mostrar=True)
    print("Red neuronal profunda en test:")


    print("Ahora corrigiendo el desbalanceo con submuestreo: ")
    X_train, X_test, y_train, y_test = pp.split(df)

    print("Regresión logística:")
    val.cross_validate(X_train, y_train, logistic_regression, folds=5, lr=True, balance=True)
    
    print("Red neuronal:")
    val.cross_validate(X_train, y_train, neural_network, folds=5, balance=True)
    
    print("Red neuronal profunda:")
    val.cross_validate(X_train, y_train, deep_neural_network, folds=5, balance=True)

    X_train_lr, X_test_lr, y_train_lr, y_test_lr = pp.preprocess(X_train, X_test, y_train, y_test, lr=True, balance=True)
    X_train, X_test, y_train, y_test = pp.preprocess(X_train, X_test, y_train, y_test, balance=True)

    # refit
    y_pred_lr = logistic_regression(X_train_lr, X_test_lr, y_train_lr)
    y_pred_nn = neural_network(X_train, X_test, y_train)

    print("Regresión logística en test:")
    ev.evaluate(y_pred_lr, y_test, mostrar=True)
    print("Red neuronal en test:")
    ev.evaluate(y_pred_nn, y_test, mostrar=True)
    print("Red neuronal profunda en test:")

    input('Pulsa Enter para cerrar el programa..')

main()
