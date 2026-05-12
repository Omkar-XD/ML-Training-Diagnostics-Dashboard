import torch
import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    accuracy_score
)

from models import get_sklearn_model, SimpleMLP
from diagnostics import detect_diagnostics
from utils import set_seed


def is_classification(y):

    unique_values = len(np.unique(y))

    # Heuristic:
    # few unique values => likely classification
    return unique_values <= 20


def train(model_type, X_train, X_val, y_train, y_val, params):

    set_seed()

    history = []

    classification_task = is_classification(y_train)

    # =========================
    # SKLEARN MODELS
    # =========================

    sklearn_models = [
        "linear_regression",
        "decision_tree_regressor",
        "logistic_regression",
        "decision_tree_classifier"
    ]

    if model_type in sklearn_models:

        model = get_sklearn_model(model_type, params)

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        # =========================
        # CLASSIFICATION METRICS
        # =========================

        if classification_task:

            train_metric = accuracy_score(
                y_train,
                train_pred
            )

            val_metric = accuracy_score(
                y_val,
                val_pred
            )

            history.append({
                "train_accuracy": float(train_metric),
                "val_accuracy": float(val_metric),
            })

        # =========================
        # REGRESSION METRICS
        # =========================

        else:

            train_metric = mean_squared_error(
                y_train,
                train_pred
            )

            val_metric = mean_squared_error(
                y_val,
                val_pred
            )

            history.append({
                "train_loss": float(train_metric),
                "val_loss": float(val_metric),
            })

    # =========================
    # MLP MODEL
    # =========================

    else:

        X_train_t = torch.tensor(
            X_train.values,
            dtype=torch.float32
        )

        X_val_t = torch.tensor(
            X_val.values,
            dtype=torch.float32
        )

        # =========================
        # CLASSIFICATION
        # =========================

        if classification_task:

            y_train_t = torch.tensor(
                y_train,
                dtype=torch.long
            )

            y_val_t = torch.tensor(
                y_val,
                dtype=torch.long
            )

            num_classes = len(np.unique(y_train))

            model = SimpleMLP(
                X_train.shape[1],
                params.get("hidden_units", 16),
            )

            # Replace final layer
            model.net[-1] = torch.nn.Linear(
                params.get("hidden_units", 16),
                num_classes
            )

            loss_fn = torch.nn.CrossEntropyLoss()

        # =========================
        # REGRESSION
        # =========================

        else:

            y_train_t = torch.tensor(
                y_train.values,
                dtype=torch.float32
            ).view(-1, 1)

            y_val_t = torch.tensor(
                y_val.values,
                dtype=torch.float32
            ).view(-1, 1)

            model = SimpleMLP(
                X_train.shape[1],
                params.get("hidden_units", 16),
            )

            loss_fn = torch.nn.MSELoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params.get("learning_rate", 0.01)
        )

        epochs = params.get("epochs", 20)

        for _ in range(epochs):

            optimizer.zero_grad()

            preds = model(X_train_t)

            loss = loss_fn(preds, y_train_t)

            loss.backward()

            optimizer.step()

            with torch.no_grad():

                val_preds = model(X_val_t)

                val_loss = loss_fn(
                    val_preds,
                    y_val_t
                )

            history.append({
                "train_loss": float(loss.item()),
                "val_loss": float(val_loss.item()),
            })

    diagnostics = detect_diagnostics(history)

    return history, diagnostics