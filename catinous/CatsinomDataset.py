from torch.utils.data.dataset import Dataset
import SimpleITK as sitk
import os
import pandas as pd
from py_jotools import augmentation, mut
import numpy as np


class CatsinomDataset(Dataset):

    def __init__(self, root_dir, datasetfile, split='train', iterations=None, batch_size=None):

        df = pd.read_csv(datasetfile)
        if type(split) is list:
            selection = np.any([df.split==x for x in split], axis=0)
        else:
            selection = df.split==split
        self.df = df.loc[selection]
        self.df = self.df.reset_index()

        if iterations is not None:
            self.df = self.df.sample(iterations*batch_size, replace=True)
            self.df = self.df.reset_index()

        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        simg = sitk.ReadImage(os.path.join(self.root_dir, self.df.iloc[index].image))
        img = sitk.GetArrayFromImage(simg)

        img = mut.intensity_window(img, low=-1024, high=400)
        img = mut.norm01(img)

        return np.tile(img, [3, 1, 1]), self.df.iloc[index].label, self.df.iloc[index].image


class Catsinom_Dataset_CatineousStream(Dataset):

    def __init__(self, root_dir, datasetfile, split='train', transition_phase_after = .8):

        df = pd.read_csv(datasetfile)
        assert(set(['train']).issubset(df.split.unique()))
        lr = df.loc[df.res=='lr']
        hr = df.loc[df.res=='hr']

        #makr sure they are random
        lr = lr.sample(len(lr))
        hr = hr.sample(len(hr))

        


        selection = df.split==split
        self.df = df.loc[selection]
        self.df = self.df.reset_index()
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        simg = sitk.ReadImage(os.path.join(self.root_dir, self.df.iloc[index].image))
        img = sitk.GetArrayFromImage(simg)

        img = mut.intensity_window(img, low=-1024, high=400)
        img = mut.norm01(img)

        return np.tile(img, [3, 1, 1]), self.df.iloc[index].label, self.df.iloc[index].image