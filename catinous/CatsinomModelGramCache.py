import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models

import pytorch_lightning as pl

from catinous.CatsinomDataset import CatsinomDataset
import argparse

class CatsinomModelGramCache(pl.LightningModule):

    def __init__(self, hparams):
        super(CatsinomModelGramCache, self).__init__()

        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Sequential(*[nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.Linear(512, 1)])

        self.loss = nn.BCEWithLogitsLoss()

        self.hparams = hparams

        self.trainingscache = {}
        self.cachematrices = {}

        self.gramlayers = [self.model.layer1[-1].conv1, self.model.layer2[-1].conv1, self.model.layer3[-1].conv1, self.model.layer4[-1].conv1]
        self.register_hooks()


    def gram_matrix(self, input):
        #taken from: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def gram_hook(self, m, input, output):
        print(m, input[0].size(), output[0].size())
        self.cachematrices[m] = self.gram_matrix(input[0])

    def register_hooks(self):
        for layer in self.gramlayers:
            layer.register_forward_hook(self.gram_hook)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, img = batch

        try:
            y_hat = self.forward(x.float())
        except ValueError as e:
            print('Value Error at', img)
            raise e

        loss = self.loss(y_hat, y[:, None].float())
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y, img = batch
        y_hat = self.forward(x.float())

        y_sig = torch.sigmoid(y_hat)
        t=torch.tensor([0.5]).to(torch.device('cuda'))
        y_sig = (y_sig > t) * 1
        acc = (y[:, None] == y_sig).float().sum()/len(y)

        return {'val_loss': self.loss(y_hat, y[:, None].float()), 'val_acc': acc}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_accuracy}
        return {'avg_val_loss': avg_loss, 'avg_val_acc': avg_accuracy, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x.float())
        y_sig = torch.sigmoid(y_hat)
        t = torch.tensor([0.5]).to(torch.device('cuda'))
        y_sig = (y_sig > t) * 1
        acc = (y[:, None] == y_sig).float().sum()/len(y)

        return {'test_loss': self.loss(y_hat, y[:, None].float()), 'test_acc': acc}

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': avg_acc}
        return {'test_loss': avg_loss, 'test_loss': avg_acc,  'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(CatsinomDataset(self.hparams.root_dir, self.hparams.datasetfile, split=self.hparams.trainsplit), shuffle=self.hparams.dataset_shuffle, batch_size=self.hparams.batch_size, num_workers=4, drop_last=True)

    @pl.data_loader
    def val_dataloader(self):
        print(self.hparams.root_dir, self.hparams.datasetfile)
        return DataLoader(CatsinomDataset(self.hparams.root_dir, self.hparams.datasetfile, split='val'), shuffle=self.hparams.dataset_shuffle, batch_size=self.hparams.batch_size, num_workers=2)

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(CatsinomDataset(self.hparams.root_dir, self.hparams.datasetfile, split='test'), batch_size=self.hparams.batch_size)
