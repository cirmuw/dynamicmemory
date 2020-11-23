import argparse
import logging
import math
import os
import random
from pprint import pprint
import numpy as np

import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.loggers as pllogging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from catinous.AgePredictor import EncoderRegressor

from catinous.CatsinomDatasetBrainAge import BrainAgeDataset, BrainAge_Continuous
from . import utils


class CatsinomModelGramCacheBrainAge(pl.LightningModule):

    def __init__(self, hparams={}, device=torch.device('cpu'), verbous=False):
        super(CatsinomModelGramCacheBrainAge, self).__init__()
        self.hparams = utils.default_params(self.get_default_hparams(), hparams)
        self.hparams = argparse.Namespace(**self.hparams)
        self.model = EncoderRegressor()

        self.learning_rate = self.hparams.learning_rate

        self.loss = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.prepareewc = False

        if not self.hparams.base_model is None:
            self.load_state_dict(torch.load(os.path.join(utils.TRAINED_MODELS_FOLDER, self.hparams.base_model), map_location=device))

            if self.hparams.EWC:
                self.prepareewc = True
                self.ewcloss = utils.BCEWithLogitWithEWCLoss(torch.Tensor([self.hparams.EWC_lambda]))

        self.to(device)

        #self.device = device

        self.shiftcheckpoint_1 = False
        self.shiftcheckpoint_2 = False

        if self.hparams.gram_weights is None:
            self.hparams.gram_weights = [1, 1, 1, 1]

        if self.hparams.use_cache and self.hparams.continous:
            self.init_cache_and_gramhooks()
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


    def init_cache_and_gramhooks(self):
        self.trainingscache = CatinousCacheAge(cachemaximum=self.hparams.cachemaximum, gram_weights=self.hparams.gram_weights)
        self.grammatrices = []
        self.gramlayers =  [self.model.encoder.feature.f_conv1_2,
                           self.model.encoder.feature.f_conv2_2,
                           self.model.encoder.feature.f_conv3_2,
                           self.model.encoder.feature.f_conv4_2]
        self.register_hooks()
        logging.info('Gram hooks and cache initialized. Cachesize: %i' % self.hparams.cachemaximum)

    @staticmethod
    def get_default_hparams():
        hparams = dict()
        hparams['root_dir'] = ''
        hparams['datasetfile'] = '/project/catinous/brainds_split.csv'
        hparams['batch_size'] = 8
        hparams['training_batch_size'] = 8
        hparams['transition_phase_after'] = 0.8
        hparams['cachemaximum'] = 128
        hparams['use_cache'] = True
        hparams['random_cache'] = False
        hparams['balance_cache'] = True
        hparams['force_misclassified'] = True
        hparams['misclassified_thresh'] = 4.0
        hparams['order'] = ['1.5T Philips', '3.0T Philips', '3.0T']
        hparams['continous'] = True
        hparams['noncontinous_steps'] = 3000
        hparams['noncontinous_train_splits'] = ['base_train']
        hparams['EWC'] = False
        hparams['EWC_dataset'] = None
        hparams['EWC_lambda'] = 1000
        hparams['EWC_bn_off'] = False
        hparams['val_check_interval'] = 100
        hparams['base_model'] = None
        hparams['run_postfix'] = '1'
        hparams['gram_weights'] = [1, 1, 1, 1]
        hparams['learning_rate'] = 0.0001

        return hparams

    def gram_matrix(self, input):
        # taken from: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
        a, b, c, d, e = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        grams = []

        for i in range(a):
            features = input[i].view(b, c * d * e)  # resise F_XL into \hat F_XL
            G = torch.mm(features, features.t())  # compute the gram product
            grams.append(G.div(b * c * d * e))

        return grams

    def gram_hook(self, m, input, output):
        self.grammatrices.append(self.gram_matrix(input[0]))

    def register_hooks(self):
        for layer in self.gramlayers:
            layer.register_forward_hook(self.gram_hook)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, filepath, res = batch
        self.grammatrices = []
        misclassified = []

        if ('1.5T Philips' in res) and ('3.0T Philips' in res): #this is not the most elegant thing to do
            exp_name = utils.get_expname_age(self.hparams)
            weights_path = utils.TRAINED_MODELS_FOLDER + exp_name + '_shift_1_ckpt.pt'
            if not self.shiftcheckpoint_1:
                torch.save(self.model.state_dict(), weights_path)
                self.shiftcheckpoint_1 = True
        elif ('3.0T Philips' in res) and ('3.0T' in res):
            exp_name = utils.get_expname_age(self.hparams)
            weights_path = utils.TRAINED_MODELS_FOLDER + exp_name + '_shift_2_ckpt.pt'
            if not self.shiftcheckpoint_2:
                torch.save(self.model.state_dict(), weights_path)
                self.shiftcheckpoint_2 = True


        if self.hparams.use_cache:
            torch.cuda.empty_cache()

            # updating the cache we are training on
            self.freeze()

            # if not self.trainingscache.cachefull:
            for ci in self.trainingscache:
                if ci is not None:
                    self.grammatrices = []
                    y_img = self.forward(ci.img.float().to(self.device))
                    grammatrix = [gm[0].cpu() for gm in self.grammatrices]
                    ci.current_grammatrix = grammatrix

            self.grammatrices = []
            y_batch = self.forward(x.float())
            batchgrammatrices = self.grammatrices
            y = y[:, None]
            for i, img in enumerate(x):
                grammatrix = [bg[i].cpu() for bg in batchgrammatrices]
                new_ci = CacheItem(img[None, :, :, :], y[i], filepath[i], res[i], y_batch[i], grammatrix)
                self.trainingscache.insert_element(new_ci)

                if self.hparams.force_misclassified:
                    if new_ci.misclassification:
                        misclassified.append(new_ci)

            self.unfreeze()

            x, y = self.trainingscache.get_training_batch(self.hparams.training_batch_size,
                                                          self.hparams.random_cache, misclassified)

            print(x.max(), x.min())

            x = x[:, None, :, :, :].to(self.device)
            y = y.to(self.device)

            y_hat = self.forward(x.float())

            loss = self.loss(y_hat, y.float())

            #tensorboard_logs = {'train_loss': loss}
            #return {'loss': loss, 'log': tensorboard_logs}
            self.log('train_loss', loss)
            return loss
        else:

            # this turns batchnorm off
            # for m in self.model.modules():
            #     if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            #         m.eval()



            if self.hparams.EWC:

                if self.prepareewc: # first tim ewc update.
                    logging.info('EWC preparation...')
                    dl = DataLoader(BrainAgeDataset(self.hparams.root_dir,
                                                    self.hparams.EWC_dataset,
                                                    iterations=100,
                                                    batch_size=8,
                                                    split=['base_train']),
                                    batch_size=8, num_workers=2, pin_memory=True)
                    self.cuda()
                    self.ewc = utils.EWC(self.model, dl)
                    # self.loss = lambda x,y,m: bc(x, y) + self.hparams.EWC_lambda * self.ewc.penalty(m)
                    logging.info('EWC preparation, done!')
                    self.prepareewc = False

                # this turns batchnorm off
                if self.hparams.EWC_bn_off:
                    for m in self.model.modules():
                        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                            m.eval()
                y_hat = self.forward(x.float())

                loss = self.ewcloss(y_hat, y[:, None].float(),self.ewc.penalty(self.model))
                # from IPython.core.debugger import set_trace
                # set_trace()
            else:
                y_hat = self.forward(x.float())
                loss = self.loss(y_hat, y[:, None].float())
            self.log('train_loss', loss)
            return loss

    def validation_step(self, batch, batch_idx):
        x, y, img, res = batch
        self.grammatrices = []

        y_hat = self.forward(x.float())

        res = res[0]

        self.log_dict({f'val_loss_{res}': self.loss(y_hat, y[:, None].float()),
                       f'val_mae_{res}': self.mae(y_hat, y[:, None].float())})


    def validation_epoch_end(self, outputs):
        val_mean = dict()
        res_count = dict()

        for output in outputs:

            for k in output.keys():
                if k not in val_mean.keys():
                    val_mean[k] = 0
                    res_count[k] = 0

                val_mean[k] += output[k]
                res_count[k] += 1

        for k in val_mean.keys():
            #tensorboard_logs[k] = val_mean[k]/res_count[k]
            self.log(k,  val_mean[k]/res_count[k])

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
        #return torch.optim.Adam(self.parameters(), lr=0.00005)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    #@pl.data_loader
    def train_dataloader(self):
        if self.hparams.continous:
            return DataLoader(BrainAge_Continuous(self.hparams.datasetfile,
                                                               transition_phase_after=self.hparams.transition_phase_after),
                              batch_size=self.hparams.batch_size, num_workers=4, drop_last=True, pin_memory=True)
        else:
            return DataLoader(BrainAgeDataset(self.hparams.datasetfile,
                                              iterations=self.hparams.noncontinous_steps,
                                              batch_size=self.hparams.batch_size,
                                              split=self.hparams.noncontinous_train_splits),
                              batch_size=self.hparams.batch_size, num_workers=4, pin_memory=True)

    #@pl.data_loader
    def val_dataloader(self):
        return DataLoader(BrainAgeDataset(self.hparams.datasetfile,
                                          split='val'),
                          batch_size=4,
                          num_workers=1, pin_memory=True)


class CacheItem():

    def __init__(self, img, label, filepath, res, current_prediction, current_grammatrix=None):
        self.img = img.detach().cpu()
        self.label = label.detach().cpu()
        self.filepath = filepath
        self.res = res
        self.traincounter = 0
        self.update_prediction(current_prediction)
        self.current_grammatrix = current_grammatrix

    def update_prediction(self, current_prediction):
        self.current_prediction = current_prediction.detach().cpu()
        self.current_loss = F.mse_loss(
            self.current_prediction, self.label.float())

        # t = torch.tensor([0.5])
        self.misclassification = abs(self.label-self.current_prediction)>4.0 #TODO: hparams misclassification thresh

    #needed for sorting the list according to current loss
    def __lt__(self, other):
        return self.current_loss < other.current_loss


class CatinousCacheAge():

    def __init__(self, cachemaximum=256, gram_weights=None):
        self.cachefull = False
        self.cachelist = []  # not sure if list is the best idea...
        self.cachemaximum = cachemaximum
        self.gram_weights = gram_weights
        self.classcounter = {0: 0, 1: 0}

    def insert_element(self, item):
        if not self.cachefull:
            self.cachelist.append(item)
            if len(self.cachelist) == self.cachemaximum:
                self.cachefull = True
        else:
            assert(item.current_grammatrix is not None)
            insertidx = -1
            mingramloss = 1000
            for j, ci in enumerate(self.cachelist):
                if ci.label==item.label:
                    l_sum = 0.0
                    for i in range(len(item.current_grammatrix)):
                        l_sum += self.gram_weights[i] * F.mse_loss(
                            item.current_grammatrix[i], ci.current_grammatrix[i], reduction='mean')

                    if l_sum < mingramloss:
                        mingramloss = l_sum
                        insertidx = j
            self.cachelist[insertidx] = item

    #forceditems are in the batch, the others are chosen randomly
    def get_training_batch(self, batchsize, randombatch=False, forceditems=None):
        batchsize = min(batchsize, len(self.cachelist))

        # TODO: read this from image and not fix it
        x = torch.empty(size=(batchsize, 64, 128, 128))
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
    model = CatsinomModelGramCacheBrainAge(hparams=hparams, device=device)
    exp_name = utils.get_expname_age(model.hparams)
    weights_path = utils.TRAINED_MODELS_FOLDER + exp_name +'.pt'
    if not os.path.exists(utils.TRAINED_MODELS_FOLDER + exp_name + '.pt'):
        logger = pllogging.TestTubeLogger(utils.LOGGING_FOLDER, name=exp_name)
        trainer = Trainer(gpus=1, max_epochs=1,
                          logger=logger, val_check_interval=model.hparams.val_check_interval,
                          checkpoint_callback=False)
        trainer.fit(model)
        model.freeze()
        torch.save(model.state_dict(), weights_path)
        if model.hparams.continous and model.hparams.use_cache:
            utils.save_cache_to_csv(model.trainingscache.cachelist, utils.TRAINED_CACHE_FOLDER + exp_name + '.csv')
    else:
        print('Read: ' + weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.freeze()

    if model.hparams.continous and model.hparams.use_cache:
        df_cache = pd.read_csv(utils.TRAINED_CACHE_FOLDER + exp_name + '.csv')

    # always get the last version
    max_version = max([int(x.split('_')[1]) for x in os.listdir(utils.LOGGING_FOLDER + exp_name)])
    logs = pd.read_csv(utils.LOGGING_FOLDER + exp_name + '/version_{}/metrics.csv'.format(max_version))

    return model, logs, df_cache, exp_name +'.pt'


def is_cached(hparams):
    model = CatsinomModelGramCacheBrainAge(hparams=hparams)
    exp_name = utils.get_expname_age(model.hparams)
    return os.path.exists(utils.TRAINED_MODELS_FOLDER + exp_name + '.pt')


def cached_path(hparams):
    model = CatsinomModelGramCacheBrainAge(hparams=hparams)
    exp_name = utils.get_expname_age(model.hparams)
    return utils.TRAINED_MODELS_FOLDER + exp_name + '.pt'
