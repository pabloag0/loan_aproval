import numpy as np
import pandas as pd
import os
from src.models import neural_network as nn
from src.models import logistic_regression as lr
from src.models import deep_neural_network as dnn
from src import preprocess as pp
from src import eda as eda
from src import evaluation as ev
from src import validation as val
from src import learning_curves as lc

def logistic_regression(X_train, X_val, y_train):
    w, b, _ = lr.train(
        X_train,
        y_train,
        np.zeros(X_train.shape[1]),
        0,
        alpha=0.1,
        num_iters=1200,
        lambda_=1
    )

    y_pred = lr.predict(X_val, w, b)

    return y_pred


def neural_network(X_train, X_val, y_train):
    
    theta1, theta2, _ = nn.train(
        X_train, 
        y_train, 
        num_labels=1, 
        alpha=0.1, 
        num_iters=1200, 
        hidden_size=8, 
        reg=0, 
        input_size=X_train.shape[1])

    y_pred = nn.predict(theta1, theta2, X_val)

    return y_pred


def deep_neural_network(X_train, X_val, y_train):
    
    model = dnn.entrenar(X_train, y_train)
    y_pred = dnn.predecir(model, X_val)

    return y_pred


def main():
    print("\nPROYECTO DE CLASIFICACION DE PRESTAMOS")
    print("---------------------------------------")
    print("\n1. CARGA DE DATOS")
    print("Cargando datos...")
    project_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(project_dir, "data", "loan_data.csv"))
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    ejecutar_eda = input("Quieres ejecutar el EDA? (s/n): ").strip().lower()
    if ejecutar_eda == "s":
        print("\n2. ANALISIS EXPLORATORIO DE DATOS")
        eda.show_dataset_info(df, plot=False)
        eda.impagos(df)
        eda.check_defaults(df)
        eda.outliers(df)
    else:
        print("EDA omitido.")

    print("\n3. SPLIT TRAIN / TEST")
    X_train, X_test, y_train, y_test = pp.split(df)
    print(f"Train: {X_train.shape[0]} ejemplos")
    print(f"Test : {X_test.shape[0]} ejemplos")

    X_train_lr, X_test_lr, y_train_lr, y_test_lr = pp.preprocess(
        X_train,
        X_test,
        y_train,
        y_test,
        lr=True
    )
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = pp.preprocess(
        X_train,
        X_test,
        y_train,
        y_test
    )

    ejecutar_curvas = input("Quieres generar las curvas? (s/n): ").strip().lower()
    if ejecutar_curvas == "s":
        print("\n4. CURVAS")
        print("Generando curva de entrenamiento de regresion logistica...")
        _, _, J_history_lr = lr.train(
            X_train_lr,
            y_train_lr,
            np.zeros(X_train_lr.shape[1]),
            0,
            alpha=0.1,
            num_iters=3000,
            lambda_=1
        )

        print("Generando curva de entrenamiento de red neuronal...")
        _, _, J_history_nn = nn.train(
            X_train_nn,
            y_train_nn,
            num_labels=1,
            alpha=0.1,
            num_iters=3000,
            hidden_size=16,
            reg=0,
            input_size=X_train_nn.shape[1]
        )

        lc.plot_training_curve(J_history_lr, "Curva de entrenamiento - Regresion logistica")
        lc.plot_training_curve(J_history_nn, "Curva de entrenamiento - Red neuronal")

        print("Generando learning curve de regresion logistica...")
        lc.learning_curve_cv(
            X_train,
            y_train,
            logistic_regression,
            folds=2,
            lr=True,
            title="Learning curve - Regresion logistica"
        )

        print("Generando learning curve de red neuronal...")
        lc.learning_curve_cv(
            X_train,
            y_train,
            neural_network,
            folds=2,
            title="Learning curve - Red neuronal"
        )

        print("Generando curva de validacion de la red neuronal...")
        lc.validation_curve_cv(
            X_train,
            y_train,
            hidden_sizes=[2, 4, 8, 16, 32],
            folds=3,
            num_iters=1200
        )
    else:
        print("Curvas omitidas.")

    print("\n5. VALIDACION CRUZADA SIN BALANCEO")
    print("Regresion logistica:")
    val.cross_validate(X_train, y_train, logistic_regression, folds=5, lr=True)

    print("Red neuronal:")
    val.cross_validate(X_train, y_train, neural_network, folds=5)

    print("Red neuronal profunda:")
    val.cross_validate(X_train, y_train, deep_neural_network, folds=5)

    print("\n6. TEST SIN BALANCEO")
    y_pred_lr = logistic_regression(X_train_lr, X_test_lr, y_train_lr)
    y_pred_nn = neural_network(X_train_nn, X_test_nn, y_train_nn)
    y_pred_dnn = deep_neural_network(X_train_nn, X_test_nn, y_train_nn)

    print("Regresion logistica:")
    ev.evaluate(y_pred_lr, y_test_lr, mostrar=True, show_matrix=True, matrix_name="test_sin_balanceo_regresion_logistica")

    print("Red neuronal:")
    ev.evaluate(y_pred_nn, y_test_nn, mostrar=True, show_matrix=True, matrix_name="test_sin_balanceo_red_neuronal")

    print("Red neuronal profunda:")
    ev.evaluate(y_pred_dnn, y_test_nn, mostrar=True, show_matrix=True, matrix_name="test_sin_balanceo_red_neuronal_profunda")

    print("\n7. VALIDACION CRUZADA CON UNDERSAMPLING")
    X_train, X_test, y_train, y_test = pp.split(df)

    print("Regresion logistica:")
    val.cross_validate(X_train, y_train, logistic_regression, folds=5, lr=True, undersampling=True)

    print("Red neuronal:")
    val.cross_validate(X_train, y_train, neural_network, folds=5, undersampling=True)

    print("Red neuronal profunda:")
    val.cross_validate(X_train, y_train, deep_neural_network, folds=5, undersampling=True)

    print("\n8. TEST CON BALANCEO")
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = pp.preprocess(
        X_train,
        X_test,
        y_train,
        y_test,
        lr=True,
        undersampling=True
    )
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = pp.preprocess(
        X_train,
        X_test,
        y_train,
        y_test,
        undersampling=True
    )

    y_pred_lr = logistic_regression(X_train_lr, X_test_lr, y_train_lr)
    y_pred_nn = neural_network(X_train_nn, X_test_nn, y_train_nn)
    y_pred_dnn = deep_neural_network(X_train_nn, X_test_nn, y_train_nn)

    print("Regresion logistica:")
    ev.evaluate(y_pred_lr, y_test_lr, mostrar=True, show_matrix=True, matrix_name="test_undersampling_regresion_logistica")

    print("Red neuronal:")
    ev.evaluate(y_pred_nn, y_test_nn, mostrar=True, show_matrix=True, matrix_name="test_undersampling_red_neuronal")

    print("Red neuronal profunda:")
    ev.evaluate(y_pred_dnn, y_test_nn, mostrar=True, show_matrix=True, matrix_name="test_undersampling_red_neuronal_profunda")

    print("\n9. VALIDACION CRUZADA CON OVERSAMPLING")
    X_train, X_test, y_train, y_test = pp.split(df)

    print("Regresion logistica:")
    val.cross_validate(X_train, y_train, logistic_regression, folds=5, lr=True, oversampling=True)

    print("Red neuronal:")
    val.cross_validate(X_train, y_train, neural_network, folds=5, oversampling=True)

    print("Red neuronal profunda:")
    val.cross_validate(X_train, y_train, deep_neural_network, folds=5, oversampling=True)

    print("\n10. TEST CON OVERSAMPLING")
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = pp.preprocess(
        X_train,
        X_test,
        y_train,
        y_test,
        lr=True,
        oversampling=True
    )
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = pp.preprocess(
        X_train,
        X_test,
        y_train,
        y_test,
        oversampling=True
    )

    y_pred_lr = logistic_regression(X_train_lr, X_test_lr, y_train_lr)
    y_pred_nn = neural_network(X_train_nn, X_test_nn, y_train_nn)
    y_pred_dnn = deep_neural_network(X_train_nn, X_test_nn, y_train_nn)

    print("Regresion logistica:")
    ev.evaluate(y_pred_lr, y_test_lr, mostrar=True, show_matrix=True, matrix_name="test_oversampling_regresion_logistica")

    print("Red neuronal:")
    ev.evaluate(y_pred_nn, y_test_nn, mostrar=True, show_matrix=True, matrix_name="test_oversampling_red_neuronal")

    print("Red neuronal profunda:")
    ev.evaluate(y_pred_dnn, y_test_nn, mostrar=True, show_matrix=True, matrix_name="test_oversampling_red_neuronal_profunda")

    print("\nFIN")
    input('Pulsa Enter para cerrar el programa..')

main()
