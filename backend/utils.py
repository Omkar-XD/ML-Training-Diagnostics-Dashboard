import json
import random
import numpy as np
import torch
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_run(run_id: str, data: dict):
    path = LOG_DIR / f"{run_id}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
