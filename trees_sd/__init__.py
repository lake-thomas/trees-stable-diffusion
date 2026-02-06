"""
Trees SD - Stable Diffusion Fine-tuning with LoRA for Tree Images

A unified package for training Stable Diffusion models (SD1.5 and SD3.5)
with LoRA on tree images from iNaturalist and Autoarborist datasets.
"""

__version__ = "0.1.0"

# Lazy imports to avoid loading heavy dependencies at import time
def __getattr__(name):
    if name == "train_model":
        from trees_sd.training.trainer import train_model
        return train_model
    elif name == "create_dataset":
        from trees_sd.datasets.loader import create_dataset
        return create_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["train_model", "create_dataset"]
