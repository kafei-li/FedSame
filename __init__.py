"""
FedSame: A Bayesian Similarity-Aware Framework for Federated Multi-Task Learning

This package implements the FedSame algorithm for federated multi-task learning
with dynamic task similarity modeling using Bayesian inference.
"""

# Import main modules for easy access
from .fedsame import train as train_fedsame
from .baselines import train as train_baselines
from .models import (
    get_mnist_model,
    get_celeba_model,
    get_ophthalmic_model,
    get_synthetic_model
)
from .data_utils import load_data
from .utils import evaluate, gini
from .config import Config

# Define what gets imported with `from package import *`
__all__ = [
    'train_fedsame',
    'train_baselines',
    'get_mnist_model',
    'get_celeba_model',
    'get_ophthalmic_model',
    'get_synthetic_model',
    'load_data',
    'evaluate',
    'gini',
    'Config'
]
