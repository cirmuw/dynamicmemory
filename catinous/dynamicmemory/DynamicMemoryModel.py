import argparse
import logging
import math
import os
import random
from pprint import pprint

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from catinous.dataset.CatsinomDataset import CatsinomDataset, Catsinom_Dataset_CatineousStream
from catinous import utils

class DynamicMemoryMoel(pl.LightningModule):

    def __init__(self, hparams={}, device=torch.device('cpu'), verbose=False):
        pass
