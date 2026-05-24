from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

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

def encode_categoricals(df: pd.DataFrame):
    """Convierte columnas de texto a nÃºmeros (label encoding o one-hot)"""
    categorical_cols = [
        "person_education",
        "person_home_ownership",
        "loan_intent",
        "previous_loan_defaults_on_file"
    ]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    return df

def handle_outliers(df):
    """Elimina valores claramente sospechosos."""
    return df[
        (df["person_age"] <= 100) &
        (df["person_emp_exp"] <= 60) &
        (df["person_income"] <= 1_000_000)
    ]

def remove_sensitive_features(df):
    df = df.drop(columns=['person_gender'])
    return df


def normalize(X_train, X_test):
    """Normaliza con z-score usando solo estadisticos de train"""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1  # Evitar division por cero

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std  

    return X_train_norm, X_test_norm

# SUBMUESTREO PARA BALANCEAR CLASES
def balance_by_loan_status(X, y, random_state=42):

    data = X.copy()
    data["loan_status"] = y

    approved = data[data["loan_status"] == 1]
    rejected = data[data["loan_status"] == 0]

    n_samples = min(len(approved), len(rejected))

    balanced_data = pd.concat([
        approved.sample(n=n_samples, random_state=random_state),
        rejected.sample(n=n_samples, random_state=random_state)
    ])

    balanced_data = balanced_data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    X_balanced = balanced_data.drop(columns=["loan_status"])
    y_balanced = balanced_data["loan_status"]

    return X_balanced, y_balanced

def preprocess(X_train, X_test, y_train, y_test, lr=False, undersampling=False, oversampling=False):

    if undersampling:
        X_train, y_train = balance_by_loan_status(X_train, y_train)

    X_train = encode_categoricals(X_train)
    X_test = encode_categoricals(X_test)

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    X_train, X_test = normalize(X_train, X_test)

    if oversampling:
        SMOTE_obj = SMOTE(random_state=42)
        X_train, y_train = SMOTE_obj.fit_resample(X_train, y_train)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    if lr:
        return X_train, X_test, y_train.ravel(), y_test.ravel()
    
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return X_train, X_test, y_train, y_test

def split(df):
    df = remove_sensitive_features(df)
    df = handle_outliers(df)

    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)
    return X_train, X_test, y_train, y_test

