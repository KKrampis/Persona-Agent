"""Save and load tensors, vectors, and JSON results."""

import json
import os
from pathlib import Path

import numpy as np
import torch


def save_tensor(tensor: torch.Tensor, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)


def load_tensor(path: str, device: str = "cpu") -> torch.Tensor:
    return torch.load(path, map_location=device)


def save_vectors(vectors: dict[str, torch.Tensor], path: str) -> None:
    """Save a dict of name->tensor as a single .pt file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in vectors.items()}, path)


def load_vectors(path: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location=device)
    return data


def save_json(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def save_numpy(arr: np.ndarray, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def load_numpy(path: str) -> np.ndarray:
    return np.load(path)


def save_projections(projections: list[float], path: str) -> None:
    save_numpy(np.array(projections), path)


def load_projections(path: str) -> np.ndarray:
    return load_numpy(path)


def results_exist(path: str) -> bool:
    return Path(path).exists()
