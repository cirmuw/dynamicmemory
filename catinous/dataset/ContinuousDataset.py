from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import nibabel as nib
import torch
from py_jotools import augmentation, mut
import pydicom as pyd
import SimpleITK as sitk
import random

class ContinuousDataset(Dataset):

    def init(self, datasetfile, transition_phase_after, order, seed):
        df = pd.read_csv(datasetfile)
        assert (set(['train']).issubset(df.split.unique()))
        np.random.seed(seed)
        print(order)
        res_dfs = list()
        for r in order:
            res_df = df.loc[df.scanner == r]
            res_df = res_df.loc[res_df.split == 'train']
            res_df = res_df.sample(frac=1, random_state=seed)

            res_dfs.append(res_df.reset_index(drop=True))

        combds = None
        new_idx = 0

        for j in range(len(res_dfs) - 1):
            old = res_dfs[j]
            new = res_dfs[j + 1]


            old_end = int((len(old) - new_idx) * transition_phase_after) + new_idx
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


class BrainAgeContinuous(ContinuousDataset):

    def __init__(self, datasetfile, transition_phase_after=.8, order=['1.5T Philips', '3.0T Philips', '3.0T'], seed=None):
        super(ContinuousDataset, self).__init__()
        self.init(datasetfile, transition_phase_after, order, seed)


    def __getitem__(self, index):
        nimg = nib.load(self.df.iloc[index].Image)
        nimg = nib.as_closest_canonical(nimg)
        img = nimg.get_fdata()
        img = img.swapaxes(0, 2)
        img = mut.resize(img, (64, 128, 128))
        img = mut.norm01(img)
        img = img[None, :, :, :]

        return torch.tensor(img).float(), torch.tensor(self.df.iloc[index].Age).float(), self.df.iloc[index].Image, self.df.iloc[index].Scanner

class LIDCContinuous(ContinuousDataset):

    def __init__(self, datasetfile, transition_phase_after=.8, order=['ges', 'geb', 'sie', 'time_siemens'], seed=None, cropped_to=(288, 288)):
        super(ContinuousDataset, self).__init__()
        self.init(datasetfile, transition_phase_after, order, seed)
        self.cropped_to = cropped_to
        self.df_multiplenodules = pd.read_csv('/project/catinous/lungnodules_allnodules.csv')

    def load_image(self, path, shiftx_aug=0, shifty_aug=0):
        # try:
        #    img = pyd.read_file(path).pixel_array
        # except Exception as e:
        #    img = pyd.read_file(path, force=True)
        #    img.file_meta.TransferSyntaxUID = pyd.uid.ImplicitVRLittleEndian
        #    img = img.pixel_array
        dcm = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(dcm)

        if self.cropped_to is not None:
            w = img.shape[1]
            s1 = int((w - self.cropped_to[0]) / 2)
            e1 = int(s1 + self.cropped_to[0])

            h = img.shape[2]
            s2 = int((h - self.cropped_to[1]) / 2)
            e2 = int(s2 + self.cropped_to[1])
            img = img[:, s1 + shiftx_aug:e1 + shiftx_aug, s2 + shifty_aug:e2 + shifty_aug]
        img = mut.intensity_window(img, low=-1024, high=1500)
        img = mut.norm01(img)

        # return img[None, :, :]
        return np.tile(img, [3, 1, 1])

    def load_annotation(self, elem, shiftx_aug=0, shifty_aug=0):
        # dcm = pyd.read_file(elem.image, force=True)
        dcm = sitk.ReadImage(elem.image)

        x = elem.x1
        y = elem.y1
        x2 = elem.x2
        y2 = elem.y2

        print(x, y, x2, y2, self.cropped_to)

        print(dcm.GetSize(), 'dcm size')
        if self.cropped_to is not None:
            x -= (dcm.GetSize()[0] - self.cropped_to[0]) / 2
            y -= (dcm.GetSize()[1] - self.cropped_to[1]) / 2
            x2 -= (dcm.GetSize()[0] - self.cropped_to[0]) / 2
            y2 -= (dcm.GetSize()[1] - self.cropped_to[1]) / 2

        y -= shiftx_aug
        x -= shifty_aug
        y2 -= shiftx_aug
        x2 -= shifty_aug

        print(x, y, x2, y2, self.cropped_to)

        xs = []
        x2s = []
        ys = []
        y2s = []
        for i, row in self.df_multiplenodules.loc[self.df_multiplenodules.image == elem.image].iterrows():

            x1_new = row.x1 - shifty_aug
            x2_new = row.x2 - shifty_aug

            y1_new = row.y1 - shiftx_aug
            y2_new = row.y2 - shiftx_aug

            if x1_new > 0 and x1_new < self.cropped_to[0] and y1_new > 0 and y1_new < self.cropped_to[1]:
                xs.append(x1_new)
                x2s.append(x2_new)

                ys.append(y1_new)
                y2s.append(y2_new)

        if xs == []:
            box = np.zeros((1, 4))
            box[0, 0] = x
            box[0, 1] = y
            box[0, 2] = x2
            box[0, 3] = y2
        else:
            box = np.zeros((len(xs) + 1, 4))
            box[0, 0] = x
            box[0, 1] = y
            box[0, 2] = x2
            box[0, 3] = y2

            for j, x in enumerate(xs):
                box[j + 1, 0] = x
                box[j + 1, 1] = ys[j]
                box[j + 1, 2] = x2s[j]
                box[j + 1, 3] = y2s[j]

        return box

    def __getitem__(self, index):
        elem = self.df.iloc[index]

        if self.cropped_to is None:
            shiftx_aug = 0
            shifty_aug = 0
        else:
            shiftx_aug = random.randint(-20, 20)
            shifty_aug = random.randint(-20, 20)

        img = self.load_image(elem.image, shiftx_aug, shifty_aug)
        annotation = self.load_annotation(elem, shiftx_aug, shifty_aug)

        target = {}
        target['boxes'] = torch.as_tensor(annotation, dtype=torch.float32)
        target['labels'] = torch.as_tensor([elem.bin_malignancy + 1] * len(annotation), dtype=torch.int64)
        target['image_id'] = torch.tensor([index] * len(annotation))

        target['area'] = torch.as_tensor(
            ((annotation[:, 3] - annotation[:, 1]) * (annotation[:, 2] - annotation[:, 0])))
        target['iscrowd'] = torch.zeros((len(annotation)), dtype=torch.int64)

        return torch.as_tensor(img, dtype=torch.float32), target, elem.scanner, elem.image

class CardiacContinuous(ContinuousDataset):

    def __init__(self, datasetfile, transition_phase_after=.8, order=['Siemens', 'GE', 'Philips', 'Canon'], seed=None):
        super(ContinuousDataset, self).__init__()
        self.init(datasetfile, transition_phase_after, order, seed)

        self.outsize = (240, 196)

    def crop_center_or_pad(self, img, cropx, cropy):
        x, y = img.shape

        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)

        if startx < 0:
            outimg = np.zeros(self.outsize)
            startx *= -1
            outimg[startx:self.outsize[0] - startx, :] = img[:, starty:starty + cropy]
            return outimg

        return img[startx:startx + cropx, starty:starty + cropy]

    def load_image(self, elem):
        #img = sitk.ReadImage(elem.filepath)
        #img = sitk.GetArrayFromImage(img)[elem.t, elem.slice, :, :]
        #img = mut.norm01(img)

        #mask = sitk.ReadImage(elem.filepath[:-7] + '_gt.nii.gz')
        #mask = sitk.GetArrayFromImage(mask)[elem.t, elem.slice, :, :]

        #if img.shape != self.outsize:
        #    img = self.crop_center_or_pad(img, self.outsize[0], self.outsize[1])
        #    mask = self.crop_center_or_pad(mask, self.outsize[0], self.outsize[1])

        img = np.load(elem.slicepath)
        mask = np.load(elem.slicepath[:-4] + '_gt.npy')

        return img[None, :, :], mask

    def __getitem__(self, index):
        elem = self.df.iloc[index]
        img, mask = self.load_image(elem)
        return torch.as_tensor(img, dtype=torch.float32), torch.as_tensor(mask,
                                                                          dtype=torch.long), elem.scanner, elem.slicepath