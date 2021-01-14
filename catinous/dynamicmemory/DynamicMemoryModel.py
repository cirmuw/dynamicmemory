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
import numpy as np
from py_jotools import augmentation, mut

from sklearn.ensemble import IsolationForest
from sklearn.random_projection import SparseRandomProjection

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
        if 'stylemodel' in self.hparams:
            self.model, self.stylemodel, self.gramlayers = utils.load_model_stylemodel(self.hparams.model)
            self.stylemodel.to(device)
            self.seperatestyle = True
        else:
            self.model, self.gramlayers = utils.load_model(self.hparams.model)
            self.seperatestyle = False

        if 'force_misclassified' in self.hparams:
            self.forcemisclassified = True
        else:
            self.forcemisclassified = False

        if 'pseudodomain_detection' in self.hparams:
            self.pseudo_detection = True
        else:
            self.pseudo_detection = False

        if not self.hparams.base_model is None:
            state_dict = torch.load(os.path.join(utils.TRAINED_MODELS_FOLDER, self.hparams.base_model))
            new_state_dict = {}
            for key in state_dict.keys():
                new_state_dict[key.replace('model.', '')] = state_dict[key]
            self.model.load_state_dict(new_state_dict)

        if self.hparams.task == 'brainage':
            self.loss = nn.MSELoss()
        elif self.hparams.task == 'lidc':
            pass
        elif self.hparams.task == 'cardiac':
            self.loss = nn.CrossEntropyLoss()

        #Initilize checkpoints to calculate BWT, FWT after training
        self.scanner_checkpoints = dict()
        self.scanner_checkpoints[self.hparams.order[0]] = True
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
        hparams['transition_phase_after'] = 0.8
        hparams['memorymaximum'] = 128
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
        hparams['gram_weights'] = None
        hparams['dim'] = 2

        return hparams

    def init_memory_and_gramhooks(self):
        if self.hparams.gram_weights is None:
            self.hparams.gram_weights = [1] * len(self.gramlayers)

        self.grammatrices = []

        for layer in self.gramlayers:
            layer.register_forward_hook(self.gram_hook)

        if self.pseudo_detection:
            base_if, base_transformer = self.get_base_domainclf()
        else:
            base_if = None
            base_transformer = None

        self.trainingsmemory = DynamicMemory(memorymaximum=self.hparams.memorymaximum,
                                                 balance_memory=self.hparams.balance_memory,
                                                 gram_weights=self.hparams.gram_weights,
                                                base_if=base_if,
                                             base_transformer=base_transformer,
                                             seed=self.hparams.seed)

        logging.info('Gram hooks and cache initialized. Memory size: %i' % self.hparams.memorymaximum)

    def get_base_domainclf(self):
        self.freeze()

        if self.hparams.task == 'cardiac':
            dl = DataLoader(CardiacBatch(self.hparams.datasetfile,
                                    split=['base'],
                                    res=self.hparams.order[0]),
                       batch_size=8, num_workers=8, drop_last=True)

        base_grams = []
        print('in base domain clf')
        for j, batch in enumerate(dl):
            self.grammatrices = []
            torch.cuda.empty_cache()

            images, targets, scanner, filepath = batch

            x = images.to(self.device)
            if self.seperatestyle:
                _ = self.stylemodel(x)
            else:
                _ = self.forward(x)

            for i, img in enumerate(x):
                grammatrix = [bg[i].cpu().detach().numpy().flatten() for bg in self.grammatrices]
                base_grams.append(np.hstack(grammatrix))

            self.grammatrices = []


        print(base_grams[0].shape)
        transformer = SparseRandomProjection(random_state=self.hparams.seed, n_components=30)
        transformer.fit(base_grams)
        trans_initelements = transformer.transform(base_grams)

        clf = IsolationForest(n_estimators=10, random_state=self.hparams.seed).fit(trans_initelements)
        self.unfreeze()
        print('finished base domain clf')
        return clf, transformer

    def gram_hook(self, m, input, output):
        if self.hparams.dim == 2:
            self.grammatrices.append(utils.gram_matrix(input[0]))
        elif self.hparams.dim == 3:
            self.grammatrices.append(utils.gram_matrix_3d(input[0]))
        else:
            raise NotImplementedError(f'gram hook with {self.hparams.dim} dimensions not defined')

    def forward(self, x):
        return self.model(x)

    def forward_lidc(self, x, y):
        return self.model(x, y)

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
                    shift_scanner = s
            if newshift:
                exp_name = utils.get_expname(self.hparams)
                weights_path = utils.TRAINED_MODELS_FOLDER + exp_name + '_shift_' + shift_scanner + '.pt'
                torch.save(self.model.state_dict(), weights_path)
                self.scanner_checkpoints[shift_scanner] = True
            self.freeze()
            #update gram matrices for current memory
            if not self.seperatestyle:
                for mi in self.trainingsmemory:
                    if mi is not None:
                        self.grammatrices = []
                        _ = self.forward(mi.img[None, :, :, :].float().to(self.device))
                        mi.current_grammatrix = [gm[0].cpu() for gm in self.grammatrices]

            #add new batch to memory
            self.grammatrices = []

            if self.seperatestyle:
                _ = self.stylemodel(x.float())
                if self.forcemisclassified:
                    y_hat = self.forward(x.float())
            else:
                y_hat = self.forward(x.float())

            if self.forcemisclassified:
                forcemetrics = []

                if self.hparams.task == 'cardiac':
                    y_hat_flat = torch.argmax(y_hat['out'], dim=1).detach().cpu().numpy()
                    y_det = y.detach().cpu().numpy()
                    for i, m in enumerate(y):
                        forcemetrics.append(mut.dice(y_det[i], y_hat_flat[i], classi=1))

            forcedelements = []
            for i, img in enumerate(x):
                grammatrix = [bg[i].cpu() for bg in self.grammatrices]
                mi = MemoryItem(img, y[i], filepath[i], scanner[i], grammatrix)
                self.trainingsmemory.insert_element(mi)
                if self.forcemisclassified:
                    if forcemetrics[i]<self.hparams.misclass_threshold:
                        forcedelements.append(mi)

            if self.pseudo_detection:
                self.trainingsmemory.check_outlier_memory(self)
                self.trainingsmemory.counter_outlier_memory()

            self.unfreeze()

            x, y = self.trainingsmemory.get_training_batch(self.hparams.batch_size, self.hparams.random_memory, forceditems=forcedelements)

            if self.hparams.task=='lidc':
                x = list(i.to(self.device) for i in x)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in y]
                loss_dict = self.forward_lidc(x, targets)
                loss = sum(l for l in loss_dict.values())
            else:
                x = x.to(self.device)
                y = torch.stack(y).to(self.device)
                y_hat = self.forward(x.float())
                loss = self.loss(y_hat['out'], y)
        else:
            if self.hparams.task == 'lidc':
                x = list(i.to(self.device) for i in x)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in y]
                loss_dict = self.forward_lidc(x, targets)
                loss = sum(l for l in loss_dict.values())
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
                                                                   transition_phase_after=self.hparams.transition_phase_after,
                                                     seed=self.hparams.seed),
                                  batch_size=self.hparams.batch_size, num_workers=8, drop_last=True)
            elif self.hparams.task == 'lidc':
                return DataLoader(LIDCContinuous(self.hparams.datasetfile,
                                                     transition_phase_after=self.hparams.transition_phase_after,
                                                 seed=self.hparams.seed),
                                  batch_size=self.hparams.batch_size, num_workers=8, collate_fn=utils.collate_fn)
            elif self.hparams.task == 'cardiac':
                return DataLoader(CardiacContinuous(self.hparams.datasetfile,
                                                 transition_phase_after=self.hparams.transition_phase_after,
                                                    seed=self.hparams.seed),
                                  batch_size=self.hparams.batch_size, num_workers=8, drop_last=True)
        else:
            if self.hparams.task == 'brainage':
                return DataLoader(BrainAgeBatch(self.hparams.datasetfile,
                                                  iterations=self.hparams.noncontinuous_steps,
                                                  batch_size=self.hparams.batch_size,
                                                  split=self.hparams.noncontinuous_train_splits),
                                  batch_size=self.hparams.batch_size, num_workers=8)
            elif self.hparams.task == 'lidc':
                return DataLoader(LIDCBatch(self.hparams.datasetfile,
                                                  iterations=self.hparams.noncontinuous_steps,
                                                  batch_size=self.hparams.batch_size,
                                                  split=self.hparams.noncontinuous_train_splits),
                                  batch_size=self.hparams.batch_size, num_workers=8, collate_fn=utils.collate_fn)
            elif self.hparams.task == 'cardiac':
                return DataLoader(CardiacBatch(self.hparams.datasetfile,
                                                  iterations=self.hparams.noncontinuous_steps,
                                                  batch_size=self.hparams.batch_size,
                                                  split=self.hparams.noncontinuous_train_splits,
                                                  res=self.hparams.scanner),
                                  batch_size=self.hparams.batch_size, num_workers=8)

    #@pl.data_loader
    def val_dataloader(self):
        if self.hparams.task == 'brainage':
            return DataLoader(BrainAgeBatch(self.hparams.datasetfile,
                                          split='val'),
                          batch_size=4,
                          num_workers=1)
        elif self.hparams.task == 'lidc':
            return DataLoader(LIDCBatch(self.hparams.datasetfile,
                                          split='val', validation=True),
                          batch_size=4,
                          num_workers=2,
                          collate_fn=utils.collate_fn)
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
        elif self.hparams.task == 'lidc':
            images, targets, scanner, filepath = batch
            images = list(image.to(self.device) for image in images)

            out = self.model(images)

            out_boxes = [
                utils.filter_boxes_area(out[i]['boxes'].cpu().detach().numpy(), out[i]['scores'].cpu().detach().numpy())
                for i in range(len(out))]

            boxes_np = [b[0] for b in out_boxes]
            scores_np = [b[1] for b in out_boxes]

            final_boxes = []
            final_scores = []
            for i, box_np in enumerate(boxes_np):
                fb, fs = utils.correct_boxes(box_np, scores_np[i])
                final_boxes.append(fb)
                final_scores.append(fs)

            gt = []
            for t in targets:
                gt.append(t['boxes'][0])

            return {'final_boxes': final_boxes, 'final_scores': final_scores, 'gt': gt, 'scanner': scanner}

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
        elif self.hparams.task == 'lidc':
            iou_thres = 0.5

            overall_true_pos = dict()
            overall_false_pos = dict()
            overall_false_neg = dict()
            overall_boxes_count = dict()
            recalls = dict()
            precision = dict()

            for scanner in self.hparams.order:
                overall_true_pos[scanner] = dict()
                overall_false_pos[scanner] = dict()
                overall_false_neg[scanner] = dict()
                overall_boxes_count[scanner] = dict()
                recalls[scanner] = []
                precision[scanner] = []
                for k in np.arange(0.0, 1.01, 0.05):
                    overall_true_pos[scanner][k] = 0
                    overall_false_pos[scanner][k] = 0
                    overall_false_neg[scanner][k] = 0
                    overall_boxes_count[scanner][k] = 0

            for out in validation_step_outputs:
                final_boxes = out['final_boxes']
                final_scores = out['final_scores']
                gt = out['gt']
                scanner = out['scanner']

                for j, fb in enumerate(final_boxes):
                    s = scanner[j]
                    g = gt[j]
                    fs = final_scores[j]

                    for k in np.arange(0.0, 1.01, 0.05):
                        false_positives = 0
                        false_negatives = 0
                        true_positives = 0
                        detected = False
                        boxes_count = 0
                        if len(fb) > 0:
                            for i, b in enumerate(fb):
                                if fs[i] > k:
                                    boxes_count += 1
                                    if utils.bb_intersection_over_union(g, b) > iou_thres:
                                        detected = True
                                    else:
                                        false_positives += 1
                            if detected:
                                true_positives += 1
                            else:
                                false_negatives += 1

                        overall_true_pos[s][k] += true_positives
                        overall_false_pos[s][k] += false_positives
                        overall_false_neg[s][k] += false_negatives
                        overall_boxes_count[s][k] += boxes_count

            aps = dict()
            for scanner in self.hparams.order:
                for k in np.arange(0.0, 1.01, 0.05):
                    if (overall_false_neg[scanner][k] + overall_true_pos[scanner][k]) == 0:
                        recalls[scanner].append(0.0)
                    else:
                        recalls[scanner].append(overall_true_pos[scanner][k] / (overall_false_neg[scanner][k] + overall_true_pos[scanner][k]))
                    if (overall_false_pos[scanner][k] + overall_true_pos[scanner][k]) == 0:
                        precision[scanner].append(0.0)
                    else:
                        precision[scanner].append(overall_true_pos[scanner][k] / (overall_false_pos[scanner][k] + overall_true_pos[scanner][k]))

                prec = np.array(precision[scanner])
                rec = np.array(recalls[scanner])
                ap = []
                for t in np.arange(0.0, 1.01, 0.1):
                    prec_arr = prec[rec > t]
                    if len(prec_arr) == 0:
                        ap.append(0.0)
                    else:
                        ap.append(prec_arr.max())
                aps[scanner] = np.array(ap).mean()

                self.log(f'val_ap_{scanner}', aps[scanner])

def trained_model(hparams, training=True):
    df_cache = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = DynamicMemoryModel(hparams=hparams, device=device)
    hparams = utils.default_params(DynamicMemoryModel.get_default_hparams(), hparams)
    exp_name = utils.get_expname(hparams)
    print(exp_name)
    weights_path = cached_path(hparams)#utils.TRAINED_MODELS_FOLDER + exp_name +'.pt'
    print('Read: ' + weights_path)

    if not os.path.exists(weights_path):
        if training:
            logger = utils.pllogger(model.hparams)
            trainer = Trainer(gpus=1, max_epochs=1, logger=logger,
                              val_check_interval=model.hparams.val_check_interval,
                              checkpoint_callback=False)
            trainer.fit(model)
            model.freeze()
            torch.save(model.state_dict(), weights_path)
            if model.hparams.continuous and model.hparams.use_memory:
                utils.save_cache_to_csv(model.trainingsmemory.memorylist, utils.TRAINED_MEMORY_FOLDER + exp_name + '.csv')
        else:
            model = None
    else:
        print('Read: ' + cached_path(hparams))
        model.load_state_dict(torch.load(cached_path(hparams), map_location=device))
        model.freeze()

    if model.hparams.continuous and model.hparams.use_memory:
        if os.path.exists(utils.TRAINED_MEMORY_FOLDER + exp_name + '.csv'):
            df_cache = pd.read_csv(utils.TRAINED_MEMORY_FOLDER + exp_name + '.csv')
        else:
            df_cache = None

    # always get the last version
    max_version = max([int(x.split('_')[1]) for x in os.listdir(utils.LOGGING_FOLDER + exp_name)])
    logs = pd.read_csv(utils.LOGGING_FOLDER + exp_name + '/version_{}/metrics.csv'.format(max_version))

    return model, logs, df_cache, exp_name +'.pt'


def is_cached(hparams):
    #model = DynamicMemoryModel(hparams=hparams)
    hparams = utils.default_params(DynamicMemoryModel.get_default_hparams(), hparams)
    exp_name = utils.get_expname(hparams)
    return os.path.exists(utils.TRAINED_MODELS_FOLDER + exp_name + '.pt')


def cached_path(hparams):
    #model = DynamicMemoryModel(hparams=hparams)
    hparams = utils.default_params(DynamicMemoryModel.get_default_hparams(), hparams)
    exp_name = utils.get_expname(hparams)
    return utils.TRAINED_MODELS_FOLDER + exp_name + '.pt'