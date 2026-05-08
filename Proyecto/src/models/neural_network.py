import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

#############################################################################################
# INFERENCIA
def predict(theta1, theta2, X):
    m = X.shape[0]

    a1 = np.hstack([np.ones((m, 1)), X]) # Ahora ya tiene el bias
    z1 = a1 @ theta1.T

    a2 = sigmoid(z1) # Pasarlo por el sigmoide

    a2 = np.hstack([np.ones((m, 1)), a2])  # De nuevo el BIAS
    z2 = a2 @ theta2.T

    a3 = sigmoid(z2) # De nuevo el sigmoide
    
    p = np.where(a3 > 0.5, 1, 0) # Salida de la clasificación binaria

    return p

#############################################################################################
# Entrenamiento

def forward_propagate(theta1, theta2, X):
    m = X.shape[0]

    a1 = np.hstack([np.ones((m, 1)), X]) # Ahora ya tiene el bias
    z1 = a1 @ theta1.T

    a2 = sigmoid(z1) # Pasarlo por el sigmoide

    a2 = np.hstack([np.ones((m, 1)), a2])  # De nuevo el BIAS
    z2 = a2 @ theta2.T

    H = sigmoid(z2) # De nuevo el sigmoide
    
    return a1, a2, H

def backprop(theta1, theta2, X, y, lambda_):

    m = X.shape[0]

    A1t, A2t, H = forward_propagate(theta1, theta2, X)
    J = cost(theta1, theta2, X, y, lambda_)

    delta3 = H - y
    delta2 = (delta3 @ theta2[:, 1:]) * A2t[:, 1:] * (1 - A2t[:, 1:])

    Delta1 = delta2.T @ A1t
    Delta2 = delta3.T @ A2t

    grad1 = Delta1 / m
    grad2 = Delta2 / m

    grad1[:, 1:] = grad1[:, 1:] + (lambda_ / m) * theta1[:, 1:]
    grad2[:, 1:] = grad2[:, 1:] + (lambda_ / m) * theta2[:, 1:]

    return (J, grad1, grad2)


def cost(theta1, theta2, X, y, lambda_):

    m = X.shape[0]

    # Propago para obtener la predicción de la red
    (a1, a2, H) = forward_propagate(theta1, theta2, X)

    epsilon = 1e-15
    H = np.clip(H, epsilon, 1 - epsilon)

    term1 = -y * np.log(H) # Primer termino
    term2 = -(1 - y) * np.log(1 - H) # Segundo termino

    J = np.sum(term1 + term2) / m

    reg_theta1 = theta1[:, 1:] # Fuera BIAS
    reg_theta2 = theta2[:, 1:] # Fuera BIAS

    reg = (lambda_ / (2 * m)) * (np.sum(reg_theta1**2) + np.sum(reg_theta2**2))

    J = J + reg
    # Pued haber una división por cero si se itera mucho, entonces hay que sumar un numero pequeño porque el log de 0 no esta definido.

    return J

def train(X, y, input_size, hidden_size, num_labels, reg, alpha, num_iters=2500): #Hidden size es el numero de neuronas que hay en la capa oculta, num_labels es el numero de clases que hay

    theta1 = (np.random.rand(hidden_size, (input_size + 1)) - 0.5) * 0.25
    theta2 = (np.random.rand(num_labels, (hidden_size + 1)) - 0.5) * 0.25

    for i in range(num_iters):
        (J, grad1, grad2) = backprop(theta1, theta2, X, y, reg)

        if i % 1000 == 0:
            print(f"Iteración {i}: Coste {J:.4f}")

        theta1 = theta1 - alpha * grad1
        theta2 = theta2 - alpha * grad2
    return theta1, theta2
