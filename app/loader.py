import torch

from .config import DATA_DIR


def load_model() -> torch.jit.ScriptModule:
    model = torch.jit.load(DATA_DIR / "model.pt")
    model.eval()
    return model
