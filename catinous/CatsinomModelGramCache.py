import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models

import pytorch_lightning as pl

from catinous.CatsinomDataset import CatsinomDataset
from catinous.CatsinomDataset import Catsinom_Dataset_CatineousStream
import random

class CatsinomModelGramCache(pl.LightningModule):

    def __init__(self, hparams, device=None):
        super(CatsinomModelGramCache, self).__init__()

        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Sequential(*[nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.Linear(512, 1)])

        self.loss = nn.BCEWithLogitsLoss()

        self.hparams = hparams

        self.trainingscache = CatinousCache(cachemaximum=self.hparams.cachemaximum)
        self.grammatrices = []

        self.device = device

        self.gramlayers = [self.model.layer1[-1].conv1, self.model.layer2[-1].conv1, self.model.layer3[-1].conv1, self.model.layer4[-1].conv1]
        self.register_hooks()


    def gram_matrix(self, input):
        #taken from: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        grams = list()

        for i in range(a):
            features = input[i].view(b, c * d)  # resise F_XL into \hat F_XL
            G = torch.mm(features, features.t())  # compute the gram product
            grams.append(G.div(b * c * d))

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

        if self.hparams.use_cache:
            torch.cuda.empty_cache()

            #updating the cache we are training on
            self.freeze()
            y_batch = self.forward(x.float())
            batchgrammatrices = self.grammatrices.copy()

            if not self.trainingscache.cachefull:
                for ci in self.trainingscache:
                    if ci is not None:
                        self.grammatrices = []
                        y_img = self.forward(ci.img.float())
                        ci.update_prediction(y_img[0])
                        grammatrix = [gm[0] for gm in self.grammatrices.copy()]
                        ci.current_grammatrix = grammatrix

            y = y[:, None]
            for i, img in enumerate(x):
                grammatrix = [bg[i] for bg in batchgrammatrices]
                self.trainingscache.insert_element(CacheItem(img[None, :, :, :], y[i], filepath[i], res[i], y_batch[i], grammatrix))

            self.unfreeze()

            x, y = self.trainingscache.get_training_batch(self.hparams.training_batch_size, self.hparams.shuffled_cache)

            x = x.to(self.device)
            y = y.to(self.device)

            y_hat = self.forward(x.float())
            loss = self.loss(y_hat, y.float())

            tensorboard_logs = {'train_loss': loss}
            return {'loss': loss, 'log': tensorboard_logs}
        else:
            y_hat = self.forward(x.float())
            loss = self.loss(y_hat, y[:, None].float())
            tensorboard_logs = {'train_loss': loss}
            return {'loss': loss, 'log': tensorboard_logs}


    def validation_step(self, batch, batch_idx):
        x, y, img, res = batch
        self.grammatrices = []

        y_hat = self.forward(x.float())

        y_sig = torch.sigmoid(y_hat)

        t=torch.tensor([0.5]).to(torch.device('cuda'))
        y_sig = (y_sig > t) * 1
        acc = (y[:, None] == y_sig).float().sum()/len(y)

        if res[0] == 'lr': #TODO: this is not completly right...
            return {'val_loss_lr': self.loss(y_hat, y[:, None].float()), 'val_acc_lr': acc}
        else:
            return {'val_loss_hr': self.loss(y_hat, y[:, None].float()), 'val_acc_hr': acc}


    def validation_end(self, outputs):

        val_loss_lr_mean = 0
        val_acc_lr_mean = 0
        val_loss_hr_mean = 0
        val_acc_hr_mean = 0
        lr_count = 0
        hr_count = 0
        for output in outputs:
            if 'val_loss_lr' in output:
                val_loss_lr_mean += output['val_loss_lr']
                val_acc_lr_mean += output['val_acc_lr']
                lr_count += 1
            else:
                val_loss_hr_mean += output['val_loss_hr']
                val_acc_hr_mean += output['val_acc_hr']
                hr_count += 1

        if lr_count>0:
            val_loss_lr_mean /= lr_count
            val_loss_lr_mean = val_loss_lr_mean.item()
            val_acc_lr_mean /= lr_count
            val_acc_lr_mean = val_acc_lr_mean.item()
        if hr_count>0:
            val_loss_hr_mean /= hr_count
            val_loss_hr_mean = val_loss_hr_mean.item()
            val_acc_hr_mean /= hr_count
            val_acc_hr_mean = val_acc_hr_mean.item()


        tensorboard_logs = {'val_loss_lr': val_loss_lr_mean, 'val_acc_lr': val_acc_lr_mean, 'val_loss_hr': val_loss_hr_mean, 'val_acc_hr': val_acc_hr_mean}
        return {'avg_val_loss_lr': val_loss_lr_mean, 'avg_val_acc_lr': val_acc_lr_mean, 'avg_val_loss_hr': val_loss_hr_mean, 'avg_val_acc_hr': val_acc_hr_mean, 'log': tensorboard_logs}

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
        return DataLoader(Catsinom_Dataset_CatineousStream(self.hparams.root_dir, self.hparams.datasetfile, transition_phase_after=self.hparams.transition_phase_after, direction=self.hparams.direction), batch_size=self.hparams.batch_size, num_workers=4)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(CatsinomDataset(self.hparams.root_dir, self.hparams.datasetfile, split='val'), batch_size=4, num_workers=2)

class CacheItem():

    def __init__(self, img, label, filepath, res, current_prediction, current_grammatrix=None):
        self.img = img
        self.label = label
        self.filepath = filepath
        self.res = res
        self.traincounter = 0
        self.current_prediction = current_prediction
        self.current_loss = F.binary_cross_entropy_with_logits(self.current_prediction, self.label.float())
        self.current_grammatrix = current_grammatrix

    def update_prediction(self, current_prediction):
        self.current_prediction = current_prediction
        self.current_loss = F.binary_cross_entropy_with_logits(self.current_prediction, self.label.float())

    #needed for sorting the list according to current loss
    def __lt__(self, other):
        return self.current_loss < other.current_loss

class CatinousCache():

    def __init__(self, cachemaximum=256):
        self.cachefull = False
        self.cachelist = list() #not sure if list is the best idea...
        self.cachemaximum = cachemaximum

    def insert_element(self, item):
        if not self.cachefull:
            self.cachelist.append(item)
            if len(self.cachelist) == self.cachemaximum:
                self.cachefull = True
        else:
            assert(item.current_grammatrix is not None)
            insertidx = -1
            mingramloss = 100
            for j, ci in enumerate(self.cachelist):
                l_sum = 0.0
                for i in range(len(item.current_grammatrix)):
                    l_sum += F.mse_loss(item.current_grammatrix[i], ci.current_grammatrix[i], reduction='sum')

                if l_sum < mingramloss:
                    mingramloss = l_sum
                    insertidx = j

            self.cachelist[insertidx] = item

    def get_training_batch(self, batchsize, randombatch=False):
        if randombatch:
            random.shuffle(self.cachelist)
        else:
            self.cachelist.sort()

        batchsize = min(batchsize, len(self.cachelist))

        x = torch.empty(size=(batchsize, 3, 512, 512)) #TODO: read this from image and not fix it
        y = torch.empty(size=(batchsize, 1))

        for j, ci in enumerate(self.cachelist[-batchsize:]):
            x[j] = ci.img
            y[j] = ci.label
            ci.traincounter += 1

        return x, y

    def __iter__(self):
        return self.cachelist.__iter__()
