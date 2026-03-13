from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import torch.nn as nn

def get_sklearn_model(model_type: str, params: dict):
    if model_type == "linear":
        return LinearRegression()
    if model_type == "tree":
        return DecisionTreeRegressor(
            max_depth=params.get("max_depth", 5),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=42
        )
    raise ValueError("Unknown model type")

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
