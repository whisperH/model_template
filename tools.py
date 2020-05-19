import numpy as np
from collections import OrderedDict
import os, copy, math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from torch.nn import functional as F

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from test_tube import HyperOptArgumentParser

from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Callback


import pytorch_lightning as pl


import os

import pandas as pd
from datetime import datetime

SEED = 7

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

root_dir = os.path.dirname(os.path.realpath(__file__))


