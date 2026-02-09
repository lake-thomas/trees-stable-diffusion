#!/usr/bin/env python3
"""
Main entry point for modular LoRA training.
Run with: python -m modular --config config.json
"""

import sys
from .train_lora import main

if __name__ == "__main__":
    main()
