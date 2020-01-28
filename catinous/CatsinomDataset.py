from torch.utils.data.dataset import Dataset
import SimpleITK as sitk
import os
import pandas as pd
from py_jotools import augmentation, mut
import numpy as np

class CatsinomDataset(Dataset):

    def __init__(self, root_dir, datasetfile, split='train'):

        df = pd.read_csv(datasetfile)
        self.df = df.loc[df.split==split]
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