import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from models import get_sklearn_model, SimpleMLP
from diagnostics import detect_diagnostics
from utils import set_seed

def train(model_type, X_train, X_val, y_train, y_val, params):
    set_seed()

    history = []

    # sklearn models
    if model_type in ["linear", "tree"]:
        model = get_sklearn_model(model_type, params)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        history.append({
            "train_loss": mean_squared_error(y_train, train_pred),
            "val_loss": mean_squared_error(y_val, val_pred),
        })

    # MLP
    else:
        X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
        y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

        model = SimpleMLP(X_train.shape[1], params.get("hidden_units", 16))
        optimizer = torch.optim.Adam(model.parameters(), lr=params.get("learning_rate", 0.01))
        loss_fn = torch.nn.MSELoss()

        for _ in range(params.get("epochs", 20)):
            optimizer.zero_grad()
            preds = model(X_train_t)
            loss = loss_fn(preds, y_train_t)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                val_loss = loss_fn(model(X_val_t), y_val_t)

            history.append({
                "train_loss": loss.item(),
                "val_loss": val_loss.item(),
            })

    diagnostics = detect_diagnostics(history)
    return history, diagnostics
