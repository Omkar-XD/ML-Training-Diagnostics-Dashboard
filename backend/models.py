from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier
)

import torch.nn as nn


def get_sklearn_model(model_type: str, params: dict):

    # =========================
    # REGRESSION MODELS
    # =========================

    if model_type == "linear_regression":

        return LinearRegression()

    if model_type == "decision_tree_regressor":

        return DecisionTreeRegressor(
            max_depth=params.get("max_depth", 5),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=42
        )

    # =========================
    # CLASSIFICATION MODELS
    # =========================

    if model_type == "logistic_regression":

        return LogisticRegression(
            max_iter=1000
        )

    if model_type == "decision_tree_classifier":

        return DecisionTreeClassifier(
            max_depth=params.get("max_depth", 5),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=42
        )

    raise ValueError(f"Unknown model type: {model_type}")


class SimpleMLP(nn.Module):

    def __init__(self, input_dim, hidden_units=16):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1)
        )

    def forward(self, x):

        return self.net(x)