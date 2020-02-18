from pytorch_lightning import Trainer
from catinous.CatsinomModelGramCache import CatsinomModelGramCache
import catinous.CatsinomModelGramCache as catsmodel
from catinous import utils as cutils
from catinous import CatsinomDataset

import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import os
import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F
import argparse
import pytorch_lightning as pl
import sklearn
from sklearn.metrics import confusion_matrix, auc, roc_curve
import torch
import pandas as pd
import seaborn as sns
import pickle
from py_jotools import mut, slurm
import numpy as np
import gc
import hashlib
import dill

hparams={'continous': True,
         'force_misclassified': True,
         'datasetfile': 'catsinom_combined_dataset.csv',
         'base_model': 'batch_lr_base_train_1_2d20289ac9.pt',
         'val_check_interval': 30,
         'cachemaximum': 512,
         'run_postfix': 'test1'}

model, logs, df_cache, basemodel_lr = catsmodel.trained_model(hparams, show_progress=True)
