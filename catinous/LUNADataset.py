from torch.utils.data.dataset import Dataset
import SimpleITK as sitk
import os
import pandas as pd
from py_jotools import augmentation, mut
import numpy as np
import nibabel as nib
import torch
import pydicom as pyd


class LUNADataset(Dataset):

    def __init__(self, datasetfile, split=['train'], iterations=None, batch_size=None, res=None, labelDebug=None):
        df = pd.read_csv(datasetfile, index_col=0)
        if type(split) is list:
            selection = np.any([df.split==x for x in split], axis=0)
        else:
            selection = df.split==split

        self.df = df.loc[selection]
        self.df = self.df.reset_index()

        if res is not None:
            self.df = self.df.loc[self.df.res==res]
            #self.df = self.df.reset_index()

        if labelDebug is not None:
            self.df = self.df.loc[self.df.label==labelDebug]

        if iterations is not None:
            self.df = self.df.sample(iterations*batch_size, replace=True)
            self.df = self.df.reset_index(drop=True)


    def __len__(self):
        return len(self.df)


    def load_image(self, path):
        img = pyd.read_file(path).pixel_array
        img = mut.intensity_window(img, low=-1024, high=400)
        img = mut.norm01(img)

        return np.tile(img, [3, 1, 1])


    def load_annotation(self, elem):
        dcm = pyd.read_file(elem.image)
        x = elem.coordX
        y = elem.coordY

        diameter = elem.diameter_mm
        spacing = float(dcm.PixelSpacing[0])

        x -= int((diameter / spacing) / 2)
        y -= int((diameter / spacing) / 2)

        x2 = x+int(diameter/spacing)
        y2 = y+int(diameter/spacing)

        box = np.zeros((1, 5))
        box[0, 0] = x
        box[0, 1] = y
        box[0, 2] = x2
        box[0, 3] = y2
        if diameter>0:
            box[0, 4] = 1
        else:
            box[0, 4] = 0

        return box

    def __getitem__(self, index):
        elem = self.df.iloc[index]
        img = self.load_image(elem.image)
        annotation = self.load_annotation(elem)


        return img, annotation, elem.image, elem.res
