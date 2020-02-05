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

        return np.tile(img, [3, 1, 1]), self.df.iloc[index].label, self.df.iloc[index].image, self.df.iloc[index].res


class Catsinom_Dataset_CatineousStream(Dataset):

    def __init__(self, root_dir, datasetfile, split='train', transition_phase_after = .8, direction='lr->hr'):

        df = pd.read_csv(datasetfile)
        assert(set(['train']).issubset(df.split.unique()))
        assert(direction in ['lr->hr', 'hr->lr', 'lrcomplete->hr'])
        lr = df.loc[df.res=='lr']
        hr = df.loc[df.res=='hr']

        #makr sure they are random
        lr = lr.sample(len(lr))
        hr = hr.sample(len(hr))

        if direction == 'lr->hr':
            old = lr.loc[lr.split=='train']
            new = hr.loc[np.logical_or(hr.split=='train',hr.split=='base_train')]
        elif direction == 'lrcomplete->hr':
            old = lr.loc[np.logical_or(lr.split=='train',lr.split=='base_train')]
            new = hr.loc[np.logical_or(hr.split == 'train', hr.split == 'base_train')]
        else:
            old = hr.loc[hr.split=='train']
            new = lr.loc[np.logical_or(lr.split=='train',lr.split=='base_train')]
        
        # old cases
        old_end = int(len(old)*transition_phase_after)
        combds = old.iloc[0:old_end]
        old_idx = old_end
        old_max = len(old)-1
        new_idx = 0
        i = 0

        while old_idx<=old_max and (i/((old_max-old_end)*2) < 1):
            take_newclass = np.random.binomial(1,min(i/((old_max-old_end)*2),1))
            if take_newclass:
                combds = combds.append(new.iloc[new_idx])
                new_idx+=1
            else:
                combds = combds.append(old.iloc[old_idx])
                old_idx+=1
            i+=1
        combds = combds.append(new.iloc[new_idx+1:])
        combds.reset_index(inplace=True)
        self.df = combds
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        simg = sitk.ReadImage(os.path.join(self.root_dir, self.df.iloc[index].image))
        img = sitk.GetArrayFromImage(simg)

        img = mut.intensity_window(img, low=-1024, high=400)
        img = mut.norm01(img)

        return np.tile(img, [3, 1, 1]), self.df.iloc[index].label, self.df.iloc[index].image, self.df.iloc[index].res