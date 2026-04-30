import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout):
        super().__init__()

        layers = []
        previous_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _to_tensor_dataset(X, y):
    X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32)
    y_tensor = torch.tensor(np.asarray(y).reshape(-1, 1), dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)


def _accuracy_from_logits(logits, y, threshold=0.5):
    predictions = (torch.sigmoid(logits) >= threshold).float()
    return (predictions == y).float().mean().item()


def _precision_from_logits(logits, y, threshold=0.5):
    predictions = (torch.sigmoid(logits) >= threshold).float()
    true_positives = ((predictions == 1) & (y == 1)).sum().item()
    predicted_positives = (predictions == 1).sum().item()

    if predicted_positives == 0:
        return 0.0

    return true_positives / predicted_positives


def _best_threshold_from_logits(logits, y):
    best_threshold = 0.5
    best_accuracy = -1.0

    for threshold in np.linspace(0.05, 0.95, 181):
        accuracy = _accuracy_from_logits(logits, y, threshold)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return best_threshold, best_accuracy


def _evaluate(model, loader, criterion, device, threshold=0.5, tune_threshold=False):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_y = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            all_logits.append(logits.cpu())
            all_y.append(y_batch.cpu())

    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    selected_threshold = threshold

    if tune_threshold:
        selected_threshold, _ = _best_threshold_from_logits(logits, y)

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": _accuracy_from_logits(logits, y, selected_threshold),
        "precision": _precision_from_logits(logits, y, selected_threshold),
        "threshold": selected_threshold,
    }


def _train_config(X_train, y_train, input_size, config, device, seed):
    torch.manual_seed(seed)

    dataset = _to_tensor_dataset(X_train, y_train)
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )

    model = DeepNeuralNetwork(
        input_size=input_size,
        hidden_layers=config["hidden_layers"],
        dropout=config["dropout"]
    ).to(device)

    y_array = np.asarray(y_train).reshape(-1)
    positives = max(1, np.sum(y_array == 1))
    negatives = max(1, np.sum(y_array == 0))
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(2, config["patience"] // 3)
    )

    best_state = None
    best_val_accuracy = -1.0
    best_threshold = 0.5
    epochs_without_improvement = 0

    for _ in range(config["epochs"]):
        model.train()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        val_metrics = _evaluate(
            model,
            val_loader,
            criterion,
            device,
            tune_threshold=True
        )
        scheduler.step(val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_threshold = val_metrics["threshold"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config["patience"]:
            break

    model.load_state_dict(best_state)
    return model, _evaluate(model, val_loader, criterion, device, best_threshold), best_threshold


def ejecutar(X_train, X_test, y_train, y_test, seed=42):
    """Entrena una red neuronal profunda y muestra su rendimiento en test.

    Recibe los datos ya preprocesados, balanceados y separados en train/test.
    Prueba varias configuraciones grandes y conserva la que consigue mayor accuracy
    en una validacion interna del conjunto de entrenamiento.
    """

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = X_train.shape[1]

    configs = [
        {
            "hidden_layers": (2048, 1024, 512, 256, 128, 64),
            "dropout": 0.18,
            "lr": 0.0007,
            "weight_decay": 0.00005,
            "batch_size": 512,
            "epochs": 220,
            "patience": 25,
        },
        {
            "hidden_layers": (4096, 2048, 1024, 512, 256, 128, 64),
            "dropout": 0.22,
            "lr": 0.0004,
            "weight_decay": 0.00002,
            "batch_size": 512,
            "epochs": 260,
            "patience": 30,
        },
        {
            "hidden_layers": (1024, 1024, 512, 512, 256, 128, 64),
            "dropout": 0.12,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 256,
            "epochs": 220,
            "patience": 25,
        },
    ]

    print("Entrenando red neuronal profunda con PyTorch...")
    print(f"Dispositivo usado: {device}")

    best_model = None
    best_config = None
    best_metrics = None
    best_threshold = 0.5

    for i, config in enumerate(configs, start=1):
        model, metrics, threshold = _train_config(
            X_train=X_train,
            y_train=y_train,
            input_size=input_size,
            config=config,
            device=device,
            seed=seed + i
        )

        print(
            f"Configuracion {i}: "
            f"val_accuracy={metrics['accuracy'] * 100:.2f}% | "
            f"val_precision={metrics['precision'] * 100:.2f}% | "
            f"threshold={threshold:.2f}"
        )

        if best_metrics is None or metrics["accuracy"] > best_metrics["accuracy"]:
            best_model = model
            best_config = config
            best_metrics = metrics
            best_threshold = threshold

    test_loader = DataLoader(
        _to_tensor_dataset(X_test, y_test),
        batch_size=best_config["batch_size"],
        shuffle=False
    )

    y_array = np.asarray(y_train).reshape(-1)
    positives = max(1, np.sum(y_array == 1))
    negatives = max(1, np.sum(y_array == 0))
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    test_metrics = _evaluate(best_model, test_loader, criterion, device, best_threshold)

    print("=" * 50)
    print("RESULTADOS RED NEURONAL PROFUNDA")
    print("=" * 50)
    print(f"Arquitectura elegida: {best_config['hidden_layers']}")
    print(f"Dropout: {best_config['dropout']}")
    print(f"Learning rate: {best_config['lr']}")
    print(f"Weight decay: {best_config['weight_decay']}")
    print(f"Threshold elegido: {best_threshold:.2f}")
    print(f"Accuracy en test: {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Precision clase positiva en test: {test_metrics['precision'] * 100:.2f}%")

    return best_model, test_metrics
