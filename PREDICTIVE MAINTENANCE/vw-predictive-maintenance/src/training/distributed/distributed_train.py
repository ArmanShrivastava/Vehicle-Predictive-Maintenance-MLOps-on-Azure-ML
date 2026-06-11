#this File is for distributed parallel training across multiple machines, using torch.distributed package

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader , DistributedSampler
import mlflow 
import numpy as np
import pandas as pd
from typing import Tuple

"""
Distributed Training
import torch.distributed as dist

Provides communication primitives:

dist.init_process_group()
dist.barrier()
dist.all_reduce()

Used for synchronization between GPUs.
"""

class VehicleFaultPredictor(nn.Module):
    """
    Neural network for predicting vehicle component failures.
    Designed to work with our 128 engineered features.
    Architecture chosen for:
    - Fast inference (important for <100ms SLO)
    - Good calibration (we need probability scores, not just labels)
    - Resistance to overfitting (limited training data per vehicle)
    """
    def __init__(self , input_dim: int = 128 , dropout_rate: float = 0.3):
        super().__init__()
        