from catinous.dataset.BatchDataset import CardiacBatch
import catinous.dynamicmemory.DynamicMemoryModel as dmodel
import torch
from torch.utils.data import DataLoader
from py_jotools import augmentation, mut
import pandas as pd

def eval_cardiac(hparams, outfile):
    device = torch.device('cuda')

    dl_test = DataLoader(CardiacBatch(hparams['datasetfile'], split=['test']), batch_size=16)
    model, _, _, _ = dmodel.trained_model(hparams)
    model.to(device)
    model.eval()

    scanners = []
    dice_1 = []
    dice_2 = []
    dice_3 = []

    for batch in dl_test:
        x, y, scanner, _ = batch
        x = x.to(device)
        y_hat = model.forward(x)['out']
        y_hat_flat = torch.argmax(y_hat, dim=1).detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        for i, m in enumerate(y):
            scanners.append(scanner[i])
            dice_1.append(mut.dice(y[i], y_hat_flat[i], classi=1))
            dice_2.append(mut.dice(y[i], y_hat_flat[i], classi=2))
            dice_3.append(mut.dice(y[i], y_hat_flat[i], classi=3))

    df_results = pd.DataFrame({'scanner': scanners, 'dice_1': dice_1, 'dice_2': dice_2, 'dice_3': dice_3})
    df_results.to_csv(outfile, index=False)