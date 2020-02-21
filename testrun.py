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

# hparams={'continous': True,
#          'force_misclassified': True,
#          'datasetfile': 'catsinom_combined_dataset.csv',
#          'base_model': 'batch_lr_base_train_1_2d20289ac9.pt',
#          'val_check_interval': 30,
#          'cachemaximum': 512,
#          'run_postfix': 'test1'}
#
# model, logs, df_cache, basemodel_lr = catsmodel.trained_model(hparams, show_progress=True)

hparams={'continous':False,
         'datasetfile': 'catsinom_lr_dataset.csv',
         'noncontinous_train_splits': ['base_train'],
         'noncontinous_steps': 3000}
_, logs, df_cache, basemodel_lr = catsmodel.trained_model(hparams)

hparams={'continous': True,
         'use_cache': False,
         'datasetfile': 'catsinom_combined_hrlowshift_dataset.csv',
         'base_model': basemodel_lr,
         'EWC': True,
         'EWC_dataset': 'catsinom_lr_dataset.csv',
         'EWC_lambda': 10000,
         'EWC_bn_off': True,
         'val_check_interval': 100}

# hparams={'continous': True,
#          'force_misclassified': True,
#          'datasetfile': 'catsinom_combined_hrlowshift_dataset.csv',
#          'base_model': basemodel_lr,
#          'val_check_interval': 99,
#          'cachemaximum': 64}

model, logs, df_cache, basemodel_lr = catsmodel.trained_model(hparams, show_progress=True)

# model = CatsinomModelGramCache(hparams=hparams, device=torch.device('cuda'))
# logger = utils.pllogger(model.hparams)
# trainer = Trainer(gpus=1, max_epochs=1, early_stop_callback=False, logger=logger, val_check_interval=model.hparams.val_check_interval, show_progress_bar=True, checkpoint_callback=False)
# trainer.fit(model)