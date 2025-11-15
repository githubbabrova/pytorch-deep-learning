"""
Contains various utility functions for PyTorch model training and saving.
"""
from pathlib import Path

import torch

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)


def load_model(model: torch.nn.Module,
               model_path: str,
               device: torch.device):
    """Loads a PyTorch model from a target path.

    Args:
    model: A target PyTorch model to load the state_dict into.
    model_path: A file path to the model.
    device: The device to map the model to (e.g. "cpu" or "cuda").

    Example usage:
    load_model(model=model_0,
               model_path="models/05_going_modular_tingvgg_model.pth",
               device=device)
    """
    # Load the model state_dict()
    print(f"[INFO] Loading model from: {model_path}")
    model.load_state_dict(torch.load(f=model_path,
                                     map_location=device))
    return model  