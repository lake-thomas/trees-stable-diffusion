"""
Training module for Stable Diffusion with LoRA
"""

# Lazy imports to avoid loading heavy dependencies
def __getattr__(name):
    if name == "LoRATrainer":
        from trees_sd.training.trainer import LoRATrainer
        return LoRATrainer
    elif name == "train_model":
        from trees_sd.training.trainer import train_model
        return train_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["LoRATrainer", "train_model"]
