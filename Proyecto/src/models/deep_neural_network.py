import numpy as np
import torch
import torch.nn as nn


class DeepNeuralNetwork(nn.Module):
    """Red neuronal sencilla para clasificacion binaria."""

    def __init__(self, input_size):
        super().__init__()

        self.red = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.red(x)


def _convertir_a_tensores(X, y):
    X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32)
    y_tensor = torch.tensor(np.asarray(y).reshape(-1, 1), dtype=torch.float32)

    return X_tensor, y_tensor


def ejecutar(X_train, X_test, y_train, y_test=None, seed=42):
    """Entrena una red neuronal basica y devuelve predicciones sobre test."""

    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    X_train_tensor, y_train_tensor = _convertir_a_tensores(X_train, y_train)
    X_test_tensor = torch.tensor(np.asarray(X_test), dtype=torch.float32)
    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)

    input_size = X_train_tensor.shape[1]
    model = DeepNeuralNetwork(input_size=input_size).to(device)

    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 100

    #print("Entrenando red neuronal sencilla con PyTorch...")
    #print(f"Entradas: {input_size}")
    #print("Arquitectura: entrada -> 32 neuronas -> 16 neuronas -> 8 neuronas -> salida")

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        predictions = model(X_train_tensor)
        loss = loss_function(predictions, y_train_tensor)

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == epochs - 1:
            #print(f"Epoca {epoch + 1:3d}/{epochs}: loss = {loss.item():.4f}")
            pass

    model.eval()
    with torch.no_grad():
        probabilities = model(X_test_tensor)
        y_pred = (probabilities >= 0.5).int().cpu().numpy().reshape(-1)

    return y_pred
