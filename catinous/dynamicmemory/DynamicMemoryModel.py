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

from catinous.dataset.ContinuousDataset import *
from catinous.dataset.BatchDataset import *
from catinous import utils
from catinous.dynamicmemory.DynamicMemory import DynamicMemory, MemoryItem


class DynamicMemoryModel(pl.LightningModule):

    def __init__(self, hparams={}, device=torch.device('cpu'), verbose=False):
        super(DynamicMemoryModel, self).__init__()
        self.hparams = utils.default_params(self.get_default_hparams(), hparams)
        self.hparams = argparse.Namespace(**self.hparams)

        self.device = device

        #load model according to hparams
        self.model, self.gramlayers = utils.load_model(self.hparams.model)
        if not self.hparams.base_model is None:
            self.load_state_dict(torch.load(os.path.join(utils.TRAINED_MODELS_FOLDER, self.hparams.base_model), map_location=device))

        if self.hparams.task == 'brainage':
            self.loss = nn.MSELoss()
        elif self.hparams.task == 'lidc':
            pass
        elif self.hparams.task == 'cardiac':
            pass

        #Initilize checkpoints to calculate BWT, FWT after training
        self.scanner_checkpoints = dict()
        for scanner in self.hparams.order[1:]:
            self.scanner_checkpoints[scanner] = False

        #Initialize memory and hooks
        if self.hparams.use_cache and self.hparams.continous:
            self.init_memory_and_gramhooks()
        else:
            if verbose:
                logging.info('No continous learning, following parameters are invalidated: \n'
                             'transition_phase_after \n'
                             'cachemaximum \n'
                             'use_cache \n'
                             'random_cache \n'
                             'force_misclassified \n'
                             'order')
            self.hparams.use_cache = False

        if verbose:
            pprint(vars(self.hparams))

    @staticmethod
    def get_default_hparams():
        hparams = dict()
        hparams['root_dir'] = '/project/catinous/'
        hparams['datasetfile'] = 'catsinom_combined_dataset.csv'
        hparams['batch_size'] = 8
        hparams['training_batch_size'] = 8
        hparams['transition_phase_after'] = 0.7
        hparams['cachemaximum'] = 128
        hparams['use_memory'] = True
        hparams['random_memory'] = True
        hparams['balance_memory'] = False
        hparams['order'] = ['lr', 'hr', 'hr_ts']
        hparams['continuous'] = True
        hparams['noncontinuous_steps'] = 3000
        hparams['noncontinuous_train_splits'] = ['train','base_train']
        hparams['val_check_interval'] = 100
        hparams['base_model'] = None
        hparams['run_postfix'] = 1

        return hparams

    def init_memory_and_gramhooks(self):
        self.trainingsmemory = DynamicMemory(memorymaximum=self.hparams.memorymaximum, balance_memory=self.hparams.balance_memory, gram_weights=self.hparams.gram_weights)
        self.grammatrices = []

        for layer in self.gramlayers:
            layer.register_forward_hook(self.gram_hook)

        logging.info('Gram hooks and cache initialized. Memory size: %i' % self.hparams.memorymaximum)

    def gram_hook(self, m, input, output):
        if self.hparams.dim == 2:
            self.grammatrices.append(utils.gram_matrix(input[0]))
        elif self.hparams.dim == 3:
            self.grammatrices.append(utils.gram_matrix_3d(input[0]))
        else:
            raise NotImplementedError(f'gram hook with {self.hparams.dim} dimensions not defined')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, filepath, scanner = batch
        self.grammatrices = []

        #save checkpoint at scanner shift
        if not self.scanner_checkpoints[scanner]:
            exp_name = utils.get_expname(self.hparams)
            weights_path = utils.TRAINED_MODELS_FOLDER + exp_name + '_shift_' + scanner +'.pt'
            torch.save(self.model.state_dict(), weights_path)
            self.scanner_checkpoints[scanner] = True

        #train with memory
        if self.hparams.use_memory:
            self.freeze()
            #update gram matrices for current memory
            for mi in self.trainingsmemory:
                if mi is not None:
                    self.grammatrices = []
                    _ = self.forward(mi.img.float().to(self.device))
                    mi.current_grammatrix = [gm[0].cpu() for gm in self.grammatrices]

            #add new batch to memory
            self.grammatrices = []
            _ = self.forward(x.float())
            for i, img in enumerate(x):
                grammatrix = [bg[i].cpu() for bg in self.grammatrices]
                mi = MemoryItem(img, y[i], filepath[i], scanner[i], grammatrix)
                self.trainingsmemory.insert_element(mi)

            self.unfreeze()

            x, y = self.trainingsmemory.get_training_batch(self.hparams.batchsize, self.hparams.random_memory)
            x = x.to(self.device)
            y = y.to(self.device)

            y_hat = self.forward(x.float())
            loss = self.loss(y_hat, y.float())
        else:
            y_hat = self.forward(x.float())
            loss = self.loss(y_hat, y[:, None].float())

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.00001)

    @pl.data_loader
    def train_dataloader(self):
        if self.hparams.continous:
            if self.hparams.task == 'brainage':
                return DataLoader(BrainAgeContinuous(self.hparams.root_dir,
                                                                   self.hparams.datasetfile,
                                                                   transition_phase_after=self.hparams.transition_phase_after),
                                  batch_size=self.hparams.batch_size, num_workers=2, drop_last=True)
            elif self.hparams.task == 'lidc':
                return DataLoader(LIDCContinuous(self.hparams.root_dir,
                                                     self.hparams.datasetfile,
                                                     transition_phase_after=self.hparams.transition_phase_after),
                                  batch_size=self.hparams.batch_size, num_workers=2, drop_last=True)
            elif self.hparams.task == 'cardiac':
                return DataLoader(CardiacContinuous(self.hparams.root_dir,
                                                 self.hparams.datasetfile,
                                                 transition_phase_after=self.hparams.transition_phase_after),
                                  batch_size=self.hparams.batch_size, num_workers=2, drop_last=True)
        else:
            if self.hparams.task == 'brainage':
                return DataLoader(BrainAgeBatch(self.hparams.root_dir,
                                                  self.hparams.datasetfile,
                                                  iterations=self.hparams.noncontinous_steps,
                                                  batch_size=self.hparams.batch_size,
                                                  split=self.hparams.noncontinous_train_splits),
                                  batch_size=self.hparams.batch_size, num_workers=2)
            elif self.hparams.task == 'lidc':
                return DataLoader(LIDCBatch(self.hparams.root_dir,
                                                  self.hparams.datasetfile,
                                                  iterations=self.hparams.noncontinous_steps,
                                                  batch_size=self.hparams.batch_size,
                                                  split=self.hparams.noncontinous_train_splits),
                                  batch_size=self.hparams.batch_size, num_workers=2)
            elif self.hparams.task == 'cardiac':
                return DataLoader(CardiacBatch(self.hparams.root_dir,
                                                  self.hparams.datasetfile,
                                                  iterations=self.hparams.noncontinous_steps,
                                                  batch_size=self.hparams.batch_size,
                                                  split=self.hparams.noncontinous_train_splits),
                                  batch_size=self.hparams.batch_size, num_workers=2)

    @pl.data_loader
    def val_dataloader(self):
        if self.hparams.task == 'brainage':
            return DataLoader(BrainAgeBatch(self.hparams.root_dir,
                                          self.hparams.datasetfile,
                                          split='val'),
                          batch_size=4,
                          num_workers=1)
        elif self.hparams.task == 'lidc':
            return DataLoader(LIDCBatch(self.hparams.root_dir,
                                          self.hparams.datasetfile,
                                          split='val'),
                          batch_size=4,
                          num_workers=1)
        elif self.hparams.task == 'cardiac':
            return DataLoader(CardiacBatch(self.hparams.root_dir,
                                          self.hparams.datasetfile,
                                          split='val'),
                          batch_size=4,
                          num_workers=1)