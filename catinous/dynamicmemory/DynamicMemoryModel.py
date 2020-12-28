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

        self.to(device)

        #load model according to hparams
        self.model, self.gramlayers = utils.load_model(self.hparams.model)
        if not self.hparams.base_model is None:
            self.load_state_dict(torch.load(os.path.join(utils.TRAINED_MODELS_FOLDER, self.hparams.base_model), map_location=device))

        if self.hparams.task == 'brainage':
            self.loss = nn.MSELoss()
        elif self.hparams.task == 'lidc':
            pass
        elif self.hparams.task == 'cardiac':
            self.loss = nn.CrossEntropyLoss()

        #Initilize checkpoints to calculate BWT, FWT after training
        self.scanner_checkpoints = dict()
        for scanner in self.hparams.order[1:]:
            self.scanner_checkpoints[scanner] = False

        #Initialize memory and hooks
        if self.hparams.use_memory and self.hparams.continuous:
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
            self.hparams.use_memory = False

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
        x, y, scanner, filepath = batch
        self.grammatrices = []

        #train with memory
        if self.hparams.use_memory and self.hparams.continuous:
            # save checkpoint at scanner shift
            newshift = False
            shifts = None
            for s in scanner:
                if s != self.hparams.order[0] and not self.scanner_checkpoints[s]:
                    newshift = True
                    shifts = s
            if newshift:
                exp_name = utils.get_expname(self.hparams)
                weights_path = utils.TRAINED_MODELS_FOLDER + exp_name + '_shift_' + scanner + '.pt'
                torch.save(self.model.state_dict(), weights_path)
                self.scanner_checkpoints[shifts] = True
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
            loss = self.loss(y_hat['out'], y)

        self.grammatrices = []
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        if self.hparams.continuous:
            return torch.optim.Adam(self.parameters(), lr=0.00001)
        else:
            return torch.optim.Adam(self.parameters(), lr=0.0001)

    #@pl.data_loader
    def train_dataloader(self):
        if self.hparams.continuous:
            if self.hparams.task == 'brainage':
                return DataLoader(BrainAgeContinuous(self.hparams.datasetfile,
                                                                   transition_phase_after=self.hparams.transition_phase_after),
                                  batch_size=self.hparams.batch_size, num_workers=2, drop_last=True)
            elif self.hparams.task == 'lidc':
                return DataLoader(LIDCContinuous(self.hparams.datasetfile,
                                                     transition_phase_after=self.hparams.transition_phase_after),
                                  batch_size=self.hparams.batch_size, num_workers=2, drop_last=True)
            elif self.hparams.task == 'cardiac':
                return DataLoader(CardiacContinuous(self.hparams.datasetfile,
                                                 transition_phase_after=self.hparams.transition_phase_after),
                                  batch_size=self.hparams.batch_size, num_workers=2, drop_last=True)
        else:
            if self.hparams.task == 'brainage':
                return DataLoader(BrainAgeBatch(self.hparams.datasetfile,
                                                  iterations=self.hparams.noncontinuous_steps,
                                                  batch_size=self.hparams.batch_size,
                                                  split=self.hparams.noncontinuous_train_splits),
                                  batch_size=self.hparams.batch_size, num_workers=2)
            elif self.hparams.task == 'lidc':
                return DataLoader(LIDCBatch(self.hparams.datasetfile,
                                                  iterations=self.hparams.noncontinuous_steps,
                                                  batch_size=self.hparams.batch_size,
                                                  split=self.hparams.noncontinuous_train_splits),
                                  batch_size=self.hparams.batch_size, num_workers=2)
            elif self.hparams.task == 'cardiac':
                return DataLoader(CardiacBatch(self.hparams.datasetfile,
                                                  iterations=self.hparams.noncontinuous_steps,
                                                  batch_size=self.hparams.batch_size,
                                                  split=self.hparams.noncontinuous_train_splits,
                                                  res=self.hparams.scanner),
                                  batch_size=self.hparams.batch_size, num_workers=2)

    #@pl.data_loader
    def val_dataloader(self):
        if self.hparams.task == 'brainage':
            return DataLoader(BrainAgeBatch(self.hparams.datasetfile,
                                          split='val'),
                          batch_size=4,
                          num_workers=1)
        elif self.hparams.task == 'lidc':
            return DataLoader(LIDCBatch(self.hparams.datasetfile,
                                          split='val'),
                          batch_size=4,
                          num_workers=1)
        elif self.hparams.task == 'cardiac':
            return DataLoader(CardiacBatch(self.hparams.datasetfile,
                                          split=['val']),
                          batch_size=4,
                          num_workers=1)

    def validation_step(self, batch, batch_idx):
        self.grammatrices = []

        if self.hparams.task == 'cardiac':
            x, y, scanner, _ = batch

            scanners = []
            dice_1 = []
            dice_2 = []
            dice_3 = []

            y_hat = self.forward(x)['out']
            y_hat_flat = torch.argmax(y_hat, dim=1).detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            for i, m in enumerate(y):
                scanners.append(scanner[i])
                dice_1.append(mut.dice(y[i], y_hat_flat[i], classi=1))
                dice_2.append(mut.dice(y[i], y_hat_flat[i], classi=2))
                dice_3.append(mut.dice(y[i], y_hat_flat[i], classi=3))
            return {'scanner': scanners, 'dice_1': dice_1, 'dice_2': dice_2, 'dice_3': dice_3}

    def validation_epoch_end(self, validation_step_outputs):
        if self.hparams.task == 'cardiac':
            scanners = []
            dice_1 = []
            dice_2 = []
            dice_3 = []

            for p in validation_step_outputs:
                scanners.extend(p['scanner'])
                dice_1.extend(p['dice_1'])
                dice_2.extend(p['dice_2'])
                dice_3.extend(p['dice_3'])

            df_dice = pd.DataFrame({'scanner': scanners, 'dice_1': dice_1, 'dice_2': dice_2, 'dice_3': dice_3})
            df_mean = df_dice.groupby('scanner').mean()
            for s in df_mean.index:
                self.log(f'val_dice1_{s}', df_mean['dice_1'][s])
                self.log(f'val_dice2_{s}', df_mean['dice_2'][s])
                self.log(f'val_dice3_{s}', df_mean['dice_3'][s])

def trained_model(hparams, show_progress = False):
    df_cache = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = DynamicMemoryModel(hparams=hparams, device=device)
    exp_name = utils.get_expname(model.hparams)
    weights_path = utils.TRAINED_MODELS_FOLDER + exp_name +'.pt'
    if not os.path.exists(utils.TRAINED_MODELS_FOLDER + exp_name + '.pt'):
        logger = utils.pllogger(model.hparams)
        trainer = Trainer(gpus=1, max_epochs=1, logger=logger,
                          val_check_interval=model.hparams.val_check_interval,
                          checkpoint_callback=False)
        trainer.fit(model)
        model.freeze()
        torch.save(model.state_dict(), weights_path)
        if model.hparams.continuous and model.hparams.use_memory:
            utils.save_cache_to_csv(model.trainingscache.cachelist, utils.TRAINED_CACHE_FOLDER + exp_name + '.csv')
    else:
        print('Read: ' + weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.freeze()

    if model.hparams.continuous and model.hparams.use_memory:
        df_cache = pd.read_csv(utils.TRAINED_CACHE_FOLDER + exp_name + '.csv')

    # always get the last version
    max_version = max([int(x.split('_')[1]) for x in os.listdir(utils.LOGGING_FOLDER + exp_name)])
    logs = pd.read_csv(utils.LOGGING_FOLDER + exp_name + '/version_{}/metrics.csv'.format(max_version))

    return model, logs, df_cache, exp_name +'.pt'


def is_cached(hparams):
    model = DynamicMemoryModel(hparams=hparams)
    exp_name = utils.get_expname(model.hparams)
    return os.path.exists(utils.TRAINED_MODELS_FOLDER + exp_name + '.pt')


def cached_path(hparams):
    model = DynamicMemoryModel(hparams=hparams)
    exp_name = utils.get_expname(model.hparams)
    return utils.TRAINED_MODELS_FOLDER + exp_name + '.pt'