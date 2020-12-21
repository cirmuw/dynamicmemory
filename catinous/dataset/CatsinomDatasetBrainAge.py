from torch.utils.data.dataset import Dataset
import SimpleITK as sitk
import os
import pandas as pd
import numpy as np
import nibabel as nib
import torch
from catinous import utils

class BrainAgeDataset(Dataset):

    def __init__(self, datasetfile, split=['base_train'], iterations=None, batch_size=None, res=None):

        df = pd.read_csv(datasetfile, index_col=0)
        if type(split) is list:
            selection = np.any([df.split==x for x in split], axis=0)
        else:
            selection = df.split==split

        self.df = df.loc[selection]
        self.df = self.df.reset_index()

        if res is not None:
            self.df = self.df.loc[self.df.Scanner==res]
            #self.df = self.df.reset_index()

        if iterations is not None:
            self.df = self.df.sample(iterations*batch_size, replace=True)
            self.df = self.df.reset_index(drop=True)


    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        nimg = nib.load(self.df.iloc[index].Image)
        nimg = nib.as_closest_canonical(nimg)
        img = nimg.get_fdata()
        img = img.swapaxes(0, 2)
        img = utils.resize(img, (64, 128, 128))
        img = utils.norm01(img)
        img = img[None, :, :, :]

        return torch.tensor(img).float(), torch.tensor(self.df.iloc[index].Age).float(), self.df.iloc[index].Image, self.df.iloc[index].Scanner


class BrainAge_Continuous(Dataset):

    def __init__(self, datasetfile, transition_phase_after=.8, order=['1.5T Philips', '3.0T Philips', '3.0T']):

        df = pd.read_csv(datasetfile, index_col=0)
        assert (set(['train']).issubset(df.split.unique()))

        np.random.seed(15613056)

        res_dfs = list()
        for r in order:
            res_df = df.loc[df.Scanner == r]
            res_df = res_df.loc[res_df.split == 'train']
            res_df = res_df.sample(frac=1)

            res_dfs.append(res_df.reset_index(drop=True))

        combds = None
        new_idx = 0

        for j in range(len(res_dfs) - 1):
            old = res_dfs[j]
            new = res_dfs[j + 1]

            old_end = int((len(old) - new_idx) * transition_phase_after) + new_idx
            print(old_end)
            if combds is None:
                combds = old.iloc[:old_end]
            else:
                combds = combds.append(old.iloc[new_idx + 1:old_end])

            old_idx = old_end
            old_max = len(old) - 1
            new_idx = 0
            i = 0

            while old_idx <= old_max and (i / ((old_max - old_end) * 2) < 1):
                take_newclass = np.random.binomial(1, min(i / ((old_max - old_end) * 2), 1))
                if take_newclass:
                    combds = combds.append(new.iloc[new_idx])
                    new_idx += 1
                else:
                    combds = combds.append(old.iloc[old_idx])
                    old_idx += 1
                i += 1
            combds = combds.append(old.iloc[old_idx:])

        combds = combds.append(new.iloc[new_idx:])
        combds.reset_index(inplace=True, drop=True)
        self.df = combds

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        nimg = nib.load(self.df.iloc[index].Image)
        nimg = nib.as_closest_canonical(nimg)
        img = nimg.get_fdata()
        img = img.swapaxes(0, 2)
        img = utils.resize(img, (64, 128, 128))
        img = utils.norm01(img)
        img = img[None, :, :, :]

        return torch.tensor(img).float(), torch.tensor(self.df.iloc[index].Age).float(), self.df.iloc[index].Image, self.df.iloc[index].Scanner
