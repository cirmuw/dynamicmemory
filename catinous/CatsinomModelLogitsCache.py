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
from . import utils


class CatsinomModelLogitsCache(pl.LightningModule):

    def __init__(self, hparams={}, device=torch.device('cpu'), verbous=False):
        super(CatsinomModelLogitsCache, self).__init__()
        self.hparams = utils.default_params(self.get_default_hparams(), hparams)
        self.hparams = argparse.Namespace(**self.hparams)
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Sequential(
            *[nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.Linear(512, 1)])

        self.loss = nn.BCEWithLogitsLoss()

        if not self.hparams.base_model is None:
            self.load_state_dict(torch.load(os.path.join(utils.TRAINED_MODELS_FOLDER, self.hparams.base_model), map_location=device))


        self.device = device
        self.t = torch.tensor([0.5]).to(device)

        self.lrcheckpoint = False
        self.hrcheckpoint = False

        if self.hparams.use_cache and self.hparams.continous:
            self.init_cache_and_hooks()
        else:
            if verbous:
                logging.info('No continous learning, following parameters are invalidated: \n'
                             'transition_phase_after \n'
                             'cachemaximum \n'
                             'use_cache \n'
                             'random_cache \n'
                             'force_misclassified \n'
                             'direction')
            self.hparams.use_cache = False



        if verbous:
            pprint(vars(self.hparams))


    def init_cache_and_hooks(self):
        self.trainingscache = CatinousLogitsCache(cachemaximum=self.hparams.cachemaximum, balance_cache=self.hparams.balance_cache)
        self.logits = []
        self.logitslayer = self.model.fc[-1]
        self.logitslayer.register_forward_hook(self.logits_hook)
        logging.info('Logits hooks and cache initialized. Cachesize: %i' % self.hparams.cachemaximum)

    @staticmethod
    def get_default_hparams():
        hparams = dict()
        hparams['root_dir'] = '/project/catinous/cat_data/'
        hparams['datasetfile'] = 'catsinom_combined_dataset.csv'
        hparams['batch_size'] = 8
        hparams['training_batch_size'] = 8
        hparams['transition_phase_after'] = 0.7
        hparams['cachemaximum'] = 128
        hparams['use_cache'] = True
        hparams['random_cache'] = True
        hparams['balance_cache'] = True
        hparams['force_misclassified'] = False
        hparams['direction'] = 'lr->hr'
        hparams['continous'] = True
        hparams['noncontinous_steps'] = 3000
        hparams['noncontinous_train_splits'] = ['train','base_train']
        hparams['val_check_interval'] = 100
        hparams['base_model'] = None
        hparams['run_postfix'] = '1'
        hparams['logits'] = True
        hparams['EWC'] = False
        hparams['EWC_dataset'] = None
        hparams['EWC_lambda'] = 1000
        hparams['EWC_bn_off'] = False
        hparams['gram_weights'] = [1, 1, 1, 1]


        return hparams

    def logits_hook(self, m, input, output):
        self.logits.append(input[0])

    #def register_hooks(self):
    #    for layer in self.logitslayer:
    #        layer.register_forward_hook(self.logits_hook)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, filepath, res = batch
        self.logits = []
        misclassified = []

        if ('lr' in res) and ('hr' in res):
            exp_name = utils.get_expname(self.hparams)
            weights_path = utils.TRAINED_MODELS_FOLDER + exp_name + '_lr_checkpoint.pt'
            if not self.lrcheckpoint:
                torch.save(self.model.state_dict(), weights_path)
                self.lrcheckpoint = True
        elif ('hr' in res) and ('hr_ts'):
            exp_name = utils.get_expname(self.hparams)
            weights_path = utils.TRAINED_MODELS_FOLDER + exp_name + '_hr_checkpoint.pt'
            if not self.hrcheckpoint:
                torch.save(self.model.state_dict(), weights_path)
                self.hrcheckpoint = True


        if self.hparams.use_cache:
            torch.cuda.empty_cache()

            # updating the cache we are training on
            self.freeze()

            # if not self.trainingscache.cachefull:
            for ci in self.trainingscache:
                if ci is not None:
                    self.logits = []
                    y_img = self.forward(ci.img.float().to(self.device))
                    # ci.update_prediction(y_img[0])
                    #grammatrix = [gm[0].cpu() for gm in self.grammatrices]
                    ci.logits = self.logits[0]

            self.logits = []
            y_batch = self.forward(x.float())
            batchlogits = self.logits
            y = y[:, None]
            for i, img in enumerate(x):
                logits = batchlogits[0][i]
                new_ci = CacheItem(img[None, :, :, :], y[i], filepath[i], res[i], y_batch[i], logits)
                self.trainingscache.insert_element(new_ci)

                if self.hparams.force_misclassified:
                    if new_ci.misclassification:
                        misclassified.append(new_ci)

            self.unfreeze()

            x, y = self.trainingscache.get_training_batch(self.hparams.training_batch_size,
                                                          self.hparams.random_cache, misclassified)

            x = x.to(self.device)
            y = y.to(self.device)

            y_hat = self.forward(x.float())

            loss = self.loss(y_hat, y.float())

            tensorboard_logs = {'train_loss': loss}
            return {'loss': loss, 'log': tensorboard_logs}
        else:

            # this turns batchnorm off
            # for m in self.model.modules():
            #     if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            #         m.eval()

            y_hat = self.forward(x.float())
            loss = self.loss(y_hat, y[:, None].float())
            tensorboard_logs = {'train_loss': loss}
            return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y, img, res = batch
        self.logits = []

        y_hat = self.forward(x.float())

        y_sig = torch.sigmoid(y_hat)

        y_sig = (y_sig > self.t).long()
        acc = (y[:, None] == y_sig).float().sum() / len(y)

        if res[0] == 'lr':  # TODO: this is not completly right...
            return {'val_loss_lr': self.loss(y_hat, y[:, None].float()), 'val_acc_lr': acc}
        elif res[0] == 'hr':
            # from IPython.core.debugger import set_trace
            # set_trace()
            return {'val_loss_hr': self.loss(y_hat, y[:, None].float()), 'val_acc_hr': acc}
        else:
            return {'val_loss_hr_ts': self.loss(y_hat, y[:, None].float()), 'val_acc_hr_ts': acc}

    def validation_end(self, outputs):
        val_loss_lr_mean = 0
        val_acc_lr_mean = 0
        val_loss_hr_mean = 0
        val_acc_hr_mean = 0
        val_loss_hr_ts_mean = 0
        val_acc_hr_ts_mean = 0
        lr_count = 0
        hr_count = 0
        hr_ts_count = 0

        for output in outputs:
            if 'val_loss_lr' in output:
                val_loss_lr_mean += output['val_loss_lr']
                val_acc_lr_mean += output['val_acc_lr']
                lr_count += 1
            elif 'val_loss_hr' in output:
                val_loss_hr_mean += output['val_loss_hr']
                val_acc_hr_mean += output['val_acc_hr']
                hr_count += 1
            else:
                val_loss_hr_ts_mean += output['val_loss_hr_ts']
                val_acc_hr_ts_mean += output['val_acc_hr_ts']
                hr_ts_count += 1

        if lr_count > 0:
            val_loss_lr_mean /= lr_count
            val_loss_lr_mean = val_loss_lr_mean.item()
            val_acc_lr_mean /= lr_count
            val_acc_lr_mean = val_acc_lr_mean.item()
        if hr_count > 0:
            val_loss_hr_mean /= hr_count
            val_loss_hr_mean = val_loss_hr_mean.item()
            val_acc_hr_mean /= hr_count
            val_acc_hr_mean = val_acc_hr_mean.item()

        if hr_ts_count > 0:
            val_loss_hr_ts_mean /= hr_ts_count
            val_loss_hr_ts_mean = val_loss_hr_ts_mean.item()
            val_acc_hr_ts_mean /= hr_ts_count
            val_acc_hr_ts_mean = val_acc_hr_ts_mean.item()

        tensorboard_logs = {'val_loss_lr': val_loss_lr_mean,
                            'val_acc_lr': val_acc_lr_mean,
                            'val_loss_hr': val_loss_hr_mean,
                            'val_acc_hr': val_acc_hr_mean,
                            'val_loss_hr_ts': val_loss_hr_ts_mean,
                            'val_acc_hr_ts': val_acc_hr_ts_mean}
        return {'avg_val_loss_lr': val_loss_lr_mean,
                'avg_val_acc_lr': val_acc_lr_mean,
                'avg_val_loss_hr': val_loss_hr_mean,
                'avg_val_acc_hr': val_acc_hr_mean,
                'avg_val_loss_hr_ts': val_loss_hr_ts_mean,
                'avg_val_acc_hr_ts': val_acc_hr_ts_mean,
                'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x.float())
        y_sig = torch.sigmoid(y_hat)
        # t = torch.tensor([0.5]).to(torch.device('cuda'))
        y_sig = (y_sig > self.t) * 1
        acc = (y[:, None] == y_sig).float().sum() / len(y)

        return {'test_loss': self.loss(y_hat, y[:, None].float()), 'test_acc': acc}

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': avg_acc}
        return {'test_loss': avg_loss, 'test_acc': avg_acc, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.00005)

    @pl.data_loader
    def train_dataloader(self):
        if self.hparams.continous:
            return DataLoader(Catsinom_Dataset_CatineousStream(self.hparams.root_dir,
                                                               self.hparams.datasetfile,
                                                               transition_phase_after=self.hparams.transition_phase_after),
                              batch_size=self.hparams.batch_size, num_workers=2, drop_last=True)
        else:
            return DataLoader(CatsinomDataset(self.hparams.root_dir,
                                              self.hparams.datasetfile,
                                              iterations=self.hparams.noncontinous_steps,
                                              batch_size=self.hparams.batch_size,
                                              split=self.hparams.noncontinous_train_splits),
                              batch_size=self.hparams.batch_size, num_workers=2)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(CatsinomDataset(self.hparams.root_dir,
                                          self.hparams.datasetfile,
                                          split='val'),
                          batch_size=4,
                          num_workers=1)


class CacheItem():

    def __init__(self, img, label, filepath, res, current_prediction, current_logits=None):
        self.img = img.detach().cpu()
        self.label = label.detach().cpu()
        self.filepath = filepath
        self.res = res
        self.traincounter = 0
        self.update_prediction(current_prediction)
        self.current_logits = current_logits

    def update_prediction(self, current_prediction):
        self.current_prediction = current_prediction.detach().cpu()
        self.current_loss = F.binary_cross_entropy_with_logits(
            self.current_prediction, self.label.float())

        y_sig = torch.sigmoid(self.current_prediction)
        # t = torch.tensor([0.5])
        self.misclassification = (self.label != ((y_sig > 0.5).long()))

    #needed for sorting the list according to current loss
    def __lt__(self, other):
        return self.current_loss < other.current_loss


class CatinousLogitsCache():

    def __init__(self, cachemaximum=256, balance_cache=True):
        self.cachefull = False
        self.cachelist = []  # not sure if list is the best idea...
        self.cachemaximum = cachemaximum
        self.balance_cache = balance_cache
        self.classcounter = {0: 0, 1: 0}

    def insert_element(self, item):
        if not self.cachefull:
            if self.balance_cache:
                if self.classcounter[item.label.item()] < math.ceil(self.cachemaximum/len(self.classcounter)):
                    self.cachelist.append(item)
                    self.classcounter[item.label.item()] += 1
            else:
                self.cachelist.append(item)
            if len(self.cachelist) == self.cachemaximum:
                self.cachefull = True
        else:
            assert(item.current_logits is not None)
            insertidx = -1
            minmseloss = 1000
            for j, ci in enumerate(self.cachelist):
                if not self.balance_cache or ci.label==item.label:
                    l = F.mse_loss(item.current_logits, ci.current_logits)
                    if l < minmseloss:
                        minmseloss = l
                        insertidx = j
            self.cachelist[insertidx] = item

    #forceditems are in the batch, the others are chosen randomly
    def get_training_batch(self, batchsize, randombatch=False, forceditems=None):
        batchsize = min(batchsize, len(self.cachelist))

        # TODO: read this from image and not fix it
        x = torch.empty(size=(batchsize, 3, 512, 512))
        y = torch.empty(size=(batchsize, 1))
        j = 0

        if forceditems is not None:
            for ci in forceditems:
                x[j] = ci.img
                y[j] = ci.label
                ci.traincounter += 1
                j += 1

            batchsize -= j

        if randombatch:
            random.shuffle(self.cachelist)
        else:
            self.cachelist.sort()

        if batchsize>0:
            for ci in self.cachelist[-batchsize:]:
                x[j] = ci.img
                y[j] = ci.label
                ci.traincounter += 1
                j += 1

        return x, y

    def __iter__(self):
        return self.cachelist.__iter__()


def trained_model(hparams, show_progress = False):
    df_cache = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = CatsinomModelLogitsCache(hparams=hparams, device=device)
    exp_name = utils.get_expname(model.hparams)
    weights_path = utils.TRAINED_MODELS_FOLDER + exp_name +'.pt'
    if not os.path.exists(utils.TRAINED_MODELS_FOLDER + exp_name + '.pt'):
        logger = utils.pllogger(model.hparams)
        trainer = Trainer(gpus=1, max_epochs=1, early_stop_callback=False, logger=logger, val_check_interval=model.hparams.val_check_interval, show_progress_bar=show_progress, checkpoint_callback=False)
        trainer.fit(model)
        model.freeze()
        torch.save(model.state_dict(), weights_path)
        if model.hparams.continous and model.hparams.use_cache:
            utils.save_cache_to_csv(model.trainingscache.cachelist, utils.TRAINED_CACHE_FOLDER + exp_name + '.csv')
    else:
        print('Read: ' + weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.freeze()
        model.eval()

    if model.hparams.continous and model.hparams.use_cache:
        df_cache = pd.read_csv(utils.TRAINED_CACHE_FOLDER + exp_name + '.csv')

    # always get the last version
    max_version = max([int(x.split('_')[1]) for x in os.listdir(utils.LOGGING_FOLDER + exp_name)])
    logs = pd.read_csv(utils.LOGGING_FOLDER + exp_name + '/version_{}/metrics.csv'.format(max_version))

    return model, logs, df_cache, exp_name +'.pt'


def is_cached(hparams):
    model = CatsinomModelLogitsCache(hparams=hparams)
    exp_name = utils.get_expname(model.hparams)
    return os.path.exists(utils.TRAINED_MODELS_FOLDER + exp_name + '.pt')


def cached_path(hparams):
    model = CatsinomModelLogitsCache(hparams=hparams)
    exp_name = utils.get_expname(model.hparams)
    return utils.TRAINED_MODELS_FOLDER + exp_name + '.pt'
