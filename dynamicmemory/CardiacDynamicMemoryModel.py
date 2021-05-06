import torch.nn as nn
from dataset.BatchDataset import *
from dataset.ContinuousDataset import *
sys.path.append('../')
import utils as dmutils
from dynamicmemory.DynamicMemoryModel import DynamicMemoryModel
from torch.utils.data import DataLoader


class CardiacDynamicMemoryModel(DynamicMemoryModel):
    def __init__(self, hparams={}, modeldir = None, device=torch.device('cpu'), training=True):
        super(DynamicMemoryModel, self).__init__()
        self.init(hparams=hparams, modeldir=modeldir, device=device, training=training)

        self.collate_fn = None
        self.TaskDatasetBatch = CardiacBatch
        self.TaskDatasetContinuous = CardiacContinuous

        self.loss = nn.CrossEntropyLoss()

        if 'EWC' in self.hparams:
            self.useewc = True
            self.ewcloss = dmutils.CrossEntropyWithEWCLoss(torch.tensor([self.hparams.EWC_lambda]))
            self.ewc = self.init_ewc()
        else:
            self.useewc = False

    def init_ewc(self):
        dl = DataLoader(self.TaskDatasetBatch(self.hparams.datasetfile,
                                              split=['base'],
                                              res=self.hparams.order[0]),
                        batch_size=8, num_workers=8, drop_last=True)
        self.model.cuda()
        ewc = dmutils.EWC(self.model, dl)

        return ewc

    def force_element(self, m):
        return m < self.hparams.misclass_threshold

    def get_forcemetrics(self, y, y_hat):
        forcemetrics = []
        y_hat_flat = torch.argmax(y_hat['out'], dim=1).detach().cpu().numpy()
        y_det = y.detach().cpu().numpy()
        for i, m in enumerate(y):
            forcemetrics.append(dmutils.dice(y_det[i], y_hat_flat[i], classi=1))

        return forcemetrics

    def get_task_loss(self, x, y):
        if self.useewc:
            x = x.to(self.device)

            y_hat = self.forward(x.float())
            loss = self.ewcloss(y_hat['out'], y, self.ewc.penalty(self.model))
        else:
            x = x.to(self.device)
            if type(y) is list:
                y = torch.stack(y).to(self.device)
            y_hat = self.forward(x.float())
            loss = self.loss(y_hat['out'], y)

        return loss

    def validation_step(self, batch, batch_idx):
        self.grammatrices = []
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
            dice_1.append(dmutils.dice(y[i], y_hat_flat[i], classi=1))
            dice_2.append(dmutils.dice(y[i], y_hat_flat[i], classi=2))
            dice_3.append(dmutils.dice(y[i], y_hat_flat[i], classi=3))
        return {'scanner': scanners, 'dice_1': dice_1, 'dice_2': dice_2, 'dice_3': dice_3}

    def validation_epoch_end(self, validation_step_outputs):
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
