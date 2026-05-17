from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

# Hay que quitar la fila del genero, no aporta valor predictivo
# Hay que cambiar las categori­as por numeros, one-hot encoding
# Hay que separar el nuevo dataset, en train y test y validation
# Hay que compensar el desbalanceo (submuestreo o sobremuestreo)
# Hay que valorar/eliminar outliers

# Feature engineering
def check_nulls(df):
    """Detecta y muestra valores nulos"""
    null_counts = df.isnull().sum()
    null_columns = null_counts[null_counts > 0]

    if null_columns.empty:
        #print('No se han encontrado valores nulos en el dataset.')
        return False

    print('Se han encontrado valores nulos en las siguientes columnas:')
    print(null_columns.sort_values(ascending=False))
    print(f'Total de valores nulos: {int(null_columns.sum())}')
    return True

def handle_nulls(df):
    """Elimina o imputa valores nulos
    No hace falta, ya que el dataset 
    no tiene valores nulos"""
    pass

def encode_categoricals(df: pd.DataFrame, defaults=False):
    """Convierte columnas de texto a nÃºmeros (label encoding o one-hot)"""
    categorical_cols = [
        "person_gender",
        "person_education",
        "person_home_ownership",
        "loan_intent"
    ]

    if defaults:
        categorical_cols.append("previous_loan_defaults_on_file")

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    return df

def handle_outliers(df):
    """Detecta y maneja outliers mediante IQR"""
    #numeric_cols = df.select_dtypes(include=[np.numbers]).columns

    df = df[df['person_age'] <= 100] # Gente de más de 100 años son excepciones
    

def normalize(X_train, X_test):
    """Normaliza con z-score usando solo estadisticos de train"""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1  # Evitar division por cero

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std  

    return X_train_norm, X_test_norm

# SUBMUESTREO PARA BALANCEAR CLASES
def balance_by_loan_status(df: pd.DataFrame, random_state=42):

    # Recorta el dataset para dejar aprobados y no aprobados a partes iguales.
    if 'loan_status' not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'loan_status'.")

    approved = df[df['loan_status'] == 1]
    rejected = df[df['loan_status'] == 0]

    if approved.empty or rejected.empty:
        raise ValueError("El DataFrame debe contener ejemplos de ambas clases.")

    n_samples = min(len(approved), len(rejected))

    approved_sample = approved.sample(n=n_samples, random_state=random_state)
    rejected_sample = rejected.sample(n=n_samples, random_state=random_state)

    balanced_df = pd.concat([approved_sample, rejected_sample])
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return balanced_df

def erase_previous_defaults(X: pd.DataFrame, y: pd.DataFrame):

    mask_no_defaults = X["previous_loan_defaults_on_file"] == "No"
    X = X[mask_no_defaults]
    y = y[mask_no_defaults]

    return X, y


def preprocess(df: pd.DataFrame, split=True, lr=False, des=False, test_size=0.25, random_state=42):
    """Funcion principal que llama a todas las anteriores en orden
       y devuelve X_train, X_val, X_test, y_train, y_val, y_test"""
    

    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    if not split:

        X = X.to_numpy()
        y = y.to_numpy().reshape(-1, 1)

        X, _ = normalize(X, X)  # Normalizamos todo junto si no hay split

        if not lr:
            return X, y
        else:
            return X, y.ravel()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25, stratify=y)

    #X_train, y_train = erase_previous_defaults(X_train, y_train)
    X_train = encode_categoricals(X_train, defaults=True)
    X_test = encode_categoricals(X_test, defaults=True)

    sm = SMOTE(random_state=42)

    X_train, X_test = normalize(X_train, X_test)

    #X_train, y_train = sm.fit_resample(X_train, y_train)

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    if lr:
        return X_train, X_test, y_train.ravel(), y_test.ravel()
    
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return X_train, X_test, y_train, y_test

def preprocess_kmeans(df: pd.DataFrame):
    """Preprocesado específico para clustering"""
    # Aquí podríamos hacer un preprocesado específico para clustering, como escalar las características, eliminar outliers, etc.

    if check_nulls(df):
        handle_nulls(df)
    
    df = encode_categoricals(df)

    handle_outliers(df)

    X = df.drop(columns=['loan_status'])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, X.columns, scaler

"""
# PREPROCESADO PARA REGRESIÓN LOGÍSTICA
def preprocess_lr(df, split=False, test_size=0.25, random_state=42):
    ""Preprocesa los datos para regresion logistica.""
    if not split:
        X, y = preprocess(df, split=split, test_size=test_size, random_state=random_state)
        return X, y.ravel()

    X_train, X_test, Y_train, Y_test = preprocess(
        df,
        split=split,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, Y_train.ravel(), Y_test.ravel()
"""