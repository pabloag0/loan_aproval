import numpy as np
import copy
import math
import matplotlib.pyplot as plt

def sigmoid(z):
    
    g = 1 / (1 + np.exp(-z))
    return g

#########################################################################
# logistic regression
#
def compute_cost(X, y, w, b, lambda_=None):
    
    m, n = X.shape
    
    # 1. Calcular z = w·x + b
    z = np.dot(X, w) + b
    
    # 2. Obtener predicciones con sigmoide
    f = sigmoid(z)
    f = np.clip(f, 1e-15, 1 - 1e-15)
    
    # 3. Calcular el coste logístico
    cost = -y * np.log(f) - (1 - y) * np.log(1 - f)
    
    # 4. Sacar la media sobre todos los ejemplos
    total_cost = (1/m) * np.sum(cost)

    return total_cost

"""
def compute_gradient(X, y, w, b, lambda_=None):

    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb_ij = X[i, j] * w[j]
            z_wb += z_wb_ij
        z_wb += b
        f_wb = sigmoid(z_wb)
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        for j in range(n):
            dj_dw_ij = (f_wb - y[i]) * X[i][j]
            dj_dw[j] += dj_dw_ij
    dj_db /= m
    dj_dw /= m

    return dj_db, dj_dw
"""

def compute_gradient(X, y, w, b):
    m = X.shape[0]

    z = X @ w + b
    f = sigmoid(z)
    error = f - y

    dj_db = np.mean(error)
    dj_dw = (X.T @ error) / m

    return dj_db, dj_dw


#########################################################################
# regularized logistic regression
#
def compute_cost_reg(X, y, w, b, lambda_=1):
    m, n = X.shape
    
    # Compute the cost without regularization
    cost_without_reg = compute_cost(X, y, w, b)
    
    # Compute regularization term (use numpy array to avoid pandas Series key errors)
    reg_term = 0

    w_arr = np.asarray(w, dtype=float).ravel()
    reg_term = (lambda_ / (2 * m)) * np.sum(w_arr ** 2)
    
    total_cost = cost_without_reg + reg_term

    return total_cost


def compute_gradient_reg(X, y, w, b, lambda_):

    m = X.shape[0]
        
    dj_db, dj_dw = compute_gradient(X, y, w, b)

    w_arr = np.asarray(w, dtype=float).ravel()
    dj_dw = dj_dw + (lambda_ / m) * w_arr

    return dj_db, dj_dw


#########################################################################
# gradient descent
#
def train(X, y, w_in, b_in=0, alpha=0.01, num_iters=1000, lambda_=None):
    m = len(X)
    if lambda_ is None:
        reg = False
    else:
        reg = True

    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        if reg:
            dj_db, dj_dw = compute_gradient_reg(X, y, w, b, lambda_)
        else:
            dj_db, dj_dw = compute_gradient(X, y, w, b)

        w -= alpha * dj_dw
        b -= alpha * dj_db
    
        if i < 100000:  # prevent resource exhaustion
            if reg:
                cost = compute_cost_reg(X, y, w, b, lambda_)
            else:
                cost = compute_cost(X, y, w, b)

            J_history.append(cost)
        if i % math.ceil(num_iters / 10) == 0 or i == num_iters - 1:
            print(f"Iteraticion {i:4d}: Cost {float(J_history[-1]):8.2f}   ")


    return w, b, J_history


#########################################################################
# plot_data
#
def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
    """
    Plots the data points X and y into a new figure
    Args:
        X : (ndarray Shape (m, 2))
        y : (ndarray Shape (m,))
        pos_label: Label for positive examples
        neg_label: Label for negative examples
    """
    positive = y == 1
    negative = y == 0

    plt.plot(X[positive, 0], X[positive, 1], 'k+', label=pos_label, markersize=7, markeredgewidth=2)
    plt.plot(X[negative, 0], X[negative, 1], 'yo', label=neg_label, markersize=7)


#########################################################################
# predict
#
"""
def predict(X, w, b):
    z = np.dot(X, w) + b
    p = (sigmoid(z) >= 0.5).astype(int)

    # Regla determinista: si la última columna es 1 → predicción 0
    mask = X.iloc[:, -1] == 1
    p[mask] = 0

    return p
"""
def predict(X, w, b):

    z = np.dot(X, w) + b
    p = sigmoid(z) >= 0.65

    return p.astype(int)

