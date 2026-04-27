import numpy as np
import copy
import math
import matplotlib.pyplot as plt


#########################################################################
# FALTA POR IMPLEMENTAR LA REGULARIZACIÓN *** HACER ***


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

def compute_gradient(X, y, w, b, lambda_=None):
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
    
    # Compute regularization term
    reg_term = 0
    if lambda_ != 0:
        for i in range(n):
            reg_term += w[i]**2
        reg_term *= lambda_ / (2 * m)
    
    total_cost = cost_without_reg + reg_term

    return total_cost


def compute_gradient_reg(X, y, w, b, lambda_=1):
    """
    Computes the gradient for linear regression 

    Args:
      X : (ndarray Shape (m,n))   variable such as house size 
      y : (ndarray Shape (m,))    actual value 
      w : (ndarray Shape (n,))    values of parameters of the model      
      b : (scalar)                value of parameter of the model  
      lambda_ : (scalar,float)    regularization constant
    Returns
      dj_db: (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw: (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 

    """
    
    return dj_db, dj_dw


#########################################################################
# gradient descent
#
def train(X, y, w_in, b_in, alpha, num_iters, lambda_=None):
    m = len(X)

    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient(X, y, w, b, lambda_)
        
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i < 100000:  # prevent resource exhaustion
            cost = compute_cost(X, y, w, b, lambda_)
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
def predict(X, w, b):

    z = np.dot(X, w) + b
    p = sigmoid(z) >= 0.5

    return p.astype(int)

