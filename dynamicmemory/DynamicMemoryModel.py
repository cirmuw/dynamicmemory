import argparse
import logging
import os
from pprint import pprint

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from sklearn.ensemble import IsolationForest
from sklearn.random_projection import SparseRandomProjection

from dataset.BatchDataset import *
from dataset.ContinuousDataset import *
sys.path.append('../')
import utils as dmutils
from dynamicmemory.DynamicMemory import DynamicMemory, MemoryItem
from abc import ABC, abstractmethod


class DynamicMemoryModel(pl.LightningModule, ABC):

    def init(self, hparams={}, modeldir = None, device=torch.device('cpu'), training=True):
        self.hparams = argparse.Namespace(**hparams)
        self.to(device)
        self.modeldir = modeldir

        # load model according to hparams
        self.model, self.stylemodel, self.gramlayers = dmutils.load_model_stylemodel(self.hparams.task)
        self.stylemodel.to(device)

        self.forcemisclassified = True if 'force_misclassified' in self.hparams else False
        self.pseudo_detection = True if 'pseudodomain_detection' in self.hparams else False

        if not self.hparams.base_model is None:
            print(os.path.join(modeldir, self.hparams.base_model))
            state_dict = torch.load(os.path.join(modeldir, self.hparams.base_model))
            new_state_dict = {}
            for key in state_dict.keys():
                if key.startswith('model.'):
                    new_state_dict[key.replace('model.', '')] = state_dict[key]
            self.model.load_state_dict(new_state_dict)

        if training:
            # Initilize checkpoints to calculate BWT, FWT after training
            self.scanner_checkpoints = dict()
            self.scanner_checkpoints[self.hparams.order[0]] = True
            for scanner in self.hparams.order[1:]:
                self.scanner_checkpoints[scanner] = False

        # Initialize memory and hooks
        if self.hparams.use_memory and self.hparams.continuous and training:
            self.init_memory_and_gramhooks()
        else:
            self.hparams.use_memory = False


    def init_memory_and_gramhooks(self):
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
        dl = DataLoader(self.TaskDatasetBatch(self.hparams.datasetfile,
                                              split=['base'],
                                              res=self.hparams.order[0]),
                        batch_size=8, num_workers=8, drop_last=True)

        base_grams = []
        for j, batch in enumerate(dl):
            self.grammatrices = []
            torch.cuda.empty_cache()

            images, targets, scanner, filepath = batch

            x = images.to(self.device)
            _ = self.forward(x)

            for i, img in enumerate(x):
                grammatrix = [bg[i].cpu().detach().numpy().flatten() for bg in self.grammatrices]
                base_grams.append(np.hstack(grammatrix))

            self.grammatrices = []

        transformer = SparseRandomProjection(random_state=self.hparams.seed, n_components=30)
        transformer.fit(base_grams)
        trans_initelements = transformer.transform(base_grams)

        clf = IsolationForest(n_estimators=10, random_state=self.hparams.seed).fit(trans_initelements)
        self.unfreeze()
        return clf, transformer

    def gram_hook(self, m, input, output):
        self.grammatrices.append(dmutils.gram_matrix(input[0]))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, scanner, filepath = batch
        self.grammatrices = []

        if self.hparams.continuous:
            # save checkpoint at scanner shift
            self.checkpoint_scanner_shift(scanner)

        # train with memory
        if self.hparams.use_memory and self.hparams.continuous:
            self.freeze()
            # add new batch to memory
            self.grammatrices = []
            if type(x) is list:
                imgsx = torch.stack(x)
                _ = self.stylemodel(imgsx.float())
            else:
                _ = self.stylemodel(x)

            # to force misclassified a forward pass with the frozen model is needed
            if self.forcemisclassified:
                if type(x) is list:
                    imgsx = torch.stack(x)
                    y_hat = self.forward(imgsx.float())
                else:
                    y_hat = self.forward(x.float())

                forcemetrics = self.get_forcemetrics(y, y_hat)

            forcedelements = []
            for i, img in enumerate(x):
                grammatrix = [bg[i].cpu() for bg in self.grammatrices]
                target = y[i]
                if type(target) == torch.Tensor:
                    det_target = target.detach().cpu()
                else:
                    det_target = {}
                    for k, v in target.items():
                        det_target[k] = v.detach().cpu()

                mi = MemoryItem(img.detach().cpu(), det_target, filepath[i], scanner[i], grammatrix)
                self.trainingsmemory.insert_element(mi)

                if self.forcemisclassified and self.force_element(forcemetrics[i]):
                    forcedelements.append(mi)

            # pseudo domains are used to balance the memory
            if self.pseudo_detection:
                self.trainingsmemory.check_outlier_memory(self)
                self.trainingsmemory.counter_outlier_memory()
                for i in range(len(self.trainingsmemory.isoforests)):
                    domainitems = self.trainingsmemory.get_domainitems(i)
                    counts = dict()
                    for o in self.hparams.order:
                        counts[o] = 0
                    for mi in domainitems:
                        counts[mi.scanner] += 1

            self.unfreeze()

            x, y = self.trainingsmemory.get_training_batch(self.hparams.training_batch_size, forceditems=forcedelements)

        loss = self.get_task_loss(x, y)
        self.grammatrices = []
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        if self.hparams.continuous:
            return torch.optim.Adam(self.parameters(), lr=0.00001)
        else:
            return torch.optim.Adam(self.parameters(), lr=0.0001)

    def checkpoint_scanner_shift(self, scanner):
        newshift = False
        for s in scanner:
            if s != self.hparams.order[0] and not self.scanner_checkpoints[s]:
                newshift = True
                shift_scanner = s
        if newshift:
            exp_name = dmutils.get_expname(self.hparams)
            weights_path = self.modeldir + exp_name + '_shift_' + shift_scanner + '.pt'
            torch.save(self.model.state_dict(), weights_path)
            self.scanner_checkpoints[shift_scanner] = True

    def train_dataloader(self):
        if self.hparams.continuous:
            return DataLoader(self.TaskDatasetContinuous(self.hparams.datasetfile,
                                                         transition_phase_after=self.hparams.transition_phase_after,
                                                         seed=self.hparams.seed,
                                                         order=self.hparams.order),
                              batch_size=self.hparams.batch_size, num_workers=8, drop_last=True,
                              collate_fn=self.collate_fn)
        else:
            return DataLoader(self.TaskDatasetBatch(self.hparams.datasetfile,
                                                    iterations=self.hparams.noncontinuous_steps,
                                                    batch_size=self.hparams.batch_size,
                                                    split=self.hparams.noncontinuous_train_splits,
                                                    res=self.hparams.scanner),
                              batch_size=self.hparams.batch_size, num_workers=8, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.TaskDatasetBatch(self.hparams.datasetfile,
                                                split='val', res=self.hparams.order),
                          batch_size=4,
                          num_workers=2,
                          collate_fn=self.collate_fn)


    @abstractmethod
    def force_element(self, m):
        pass

    @abstractmethod
    def get_forcemetrics(self, y, y_hat):
        pass

    @abstractmethod
    def get_task_loss(self, x, y):
        pass