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



def main():
    os.system('cls')

    print('Cargando datos...')
    df = pd.read_csv("data/loan_data.csv")


    

    input('Presiona enter para cerrar el programa')

#############################################
# ONLY DEBUGGING PURPOSES, DELETE THIS LATER#
#############################################
import sys                                  #
sys.dont_write_bytecode = True              #
#############################################

main()