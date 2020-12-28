from py_jotools import augmentation, mut
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
import nibabel as nib
import torch
import pydicom as pyd
import SimpleITK as sitk
import random

class BatchDataset(Dataset):

    def init(self, datasetfile, split, iterations, batch_size, res, seed):
        df = pd.read_csv(datasetfile, index_col=0)
        if type(split) is list:
            selection = np.any([df.split == x for x in split], axis=0)
        else:
            selection = df.split == split

        self.df = df.loc[selection]
        self.df = self.df.reset_index()

        if res is not None:
            self.df = self.df.loc[self.df.scanner == res]

        if iterations is not None:
            self.df = self.df.sample(iterations * batch_size, replace=True, random_state=seed)
            self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)


class BrainAgeBatch(BatchDataset):

    def __init__(self, datasetfile, split=['base_train'], iterations=None, batch_size=None, res=None, seed=None):
        super(BatchDataset, self).__init__()
        self.init(datasetfile, split, iterations, batch_size, res, seed)


    def __getitem__(self, index):
        nimg = nib.load(self.df.iloc[index].Image)
        nimg = nib.as_closest_canonical(nimg)
        img = nimg.get_fdata()
        img = img.swapaxes(0, 2)
        img = mut.resize(img, (64, 128, 128))
        img = mut.norm01(img)
        img = img[None, :, :, :]

        return torch.tensor(img).float(), torch.tensor(self.df.iloc[index].Age).float(), self.df.iloc[index].Image, \
               self.df.iloc[index].Scanner

class LIDCBatch(BatchDataset):

    def __init__(self, datasetfile, split=['base'], iterations=None, batch_size=None, res=None, seed=None,
                 cropped_to=None, validation=False):
        super(BatchDataset, self).__init__()
        self.init(datasetfile, split, iterations, batch_size, res, seed)


        self.cropped_to = cropped_to
        self.validation = validation

    def load_image(self, path, shiftx_aug=0, shifty_aug=0):
        img = pyd.read_file(path).pixel_array
        if self.cropped_to is not None:
            w = img.shape[0]
            s1 = int((w - self.cropped_to[0]) / 2)
            e1 = int(s1 + self.cropped_to[0])

            h = img.shape[1]
            s2 = int((h - self.cropped_to[1]) / 2)
            e2 = int(s2 + self.cropped_to[1])
            img = img[s1 + shiftx_aug:e1 + shiftx_aug, s2 + shifty_aug:e2 + shifty_aug]
        img = mut.intensity_window(img, low=-1024, high=1500)
        img = mut.norm01(img)

        # return img[None, :, :]
        return np.tile(img, [3, 1, 1])

    def load_image_validation(self, path):
        img = pyd.read_file(path).pixel_array

        imgs = []

        w = img.shape[0]
        h = img.shape[1]
        x_shift = int((w - self.cropped_to[0]) / 2)
        y_shift = int((h - self.cropped_to[1]) / 2)
        s1 = x_shift
        e1 = int(s1 + self.cropped_to[0])
        s2 = y_shift
        e2 = int(s2 + self.cropped_to[1])

        im_crop = img[s1:e1, s2:e2]
        im_crop = mut.intensity_window(im_crop, low=-1024, high=1500)
        im_crop = mut.norm01(im_crop)
        imgs.append(np.tile(im_crop, [3, 1, 1]))  # center crop
        im_crop = img[s1 - x_shift:e1 - x_shift, s2 - y_shift:e2 - y_shift]
        im_crop = mut.intensity_window(im_crop, low=-1024, high=1500)
        im_crop = mut.norm01(im_crop)
        imgs.append(np.tile(im_crop, [3, 1, 1]))  # center crop
        im_crop = img[s1 + x_shift:e1 + x_shift, s2 - y_shift:e2 - y_shift]
        im_crop = mut.intensity_window(im_crop, low=-1024, high=1500)
        im_crop = mut.norm01(im_crop)
        imgs.append(np.tile(im_crop, [3, 1, 1]))  # center crop
        im_crop = img[s1 - x_shift:e1 - x_shift, s2 + y_shift:e2 + y_shift]
        im_crop = mut.intensity_window(im_crop, low=-1024, high=1500)
        im_crop = mut.norm01(im_crop)
        imgs.append(np.tile(im_crop, [3, 1, 1]))  # center crop
        im_crop = img[s1 + x_shift:e1 + x_shift, s2 + y_shift:e2 + y_shift]
        im_crop = mut.intensity_window(im_crop, low=-1024, high=1500)
        im_crop = mut.norm01(im_crop)
        imgs.append(np.tile(im_crop, [3, 1, 1]))  # center crop

        return np.array(imgs)

    def load_annotation(self, elem, shiftx_aug=0, shifty_aug=0, validation=False):
        dcm = pyd.read_file(elem.image)
        x = elem.coordX
        y = elem.coordY
        diameter = elem.diameter_mm
        spacing = float(dcm.PixelSpacing[0])

        if not validation:
            if self.cropped_to is not None:
                x -= (dcm.Rows - self.cropped_to[0]) / 2
                y -= (dcm.Columns - self.cropped_to[1]) / 2
            y -= shiftx_aug
            x -= shifty_aug

        x -= int((diameter / spacing) / 2)
        y -= int((diameter / spacing) / 2)

        x2 = x + int(diameter / spacing)
        y2 = y + int(diameter / spacing)

        box = np.zeros((1, 4))
        box[0, 0] = x
        box[0, 1] = y
        box[0, 2] = x2
        box[0, 3] = y2

        return box

    def __getitem__(self, index):
        elem = self.df.iloc[index]

        if not self.validation:
            if self.cropped_to is None:
                shiftx_aug = 0
                shifty_aug = 0
            else:
                shiftx_aug = random.randint(-20, 20)
                shifty_aug = random.randint(-20, 20)

            img = self.load_image(elem.image, shiftx_aug, shifty_aug)
            annotation = self.load_annotation(elem, shiftx_aug, shifty_aug)
        else:
            img = self.load_image_validation(elem.image)
            annotation = self.load_annotation(elem, 0, 0, True)

        target = {}
        target['boxes'] = torch.as_tensor(annotation, dtype=torch.float32)
        target['labels'] = torch.as_tensor((elem.bin_malignancy + 1,), dtype=torch.int64)
        target['image_id'] = torch.tensor([index])
        target['area'] = torch.as_tensor(
            ((annotation[:, 3] - annotation[:, 1]) * (annotation[:, 2] - annotation[:, 0])))
        target['iscrowd'] = torch.zeros((1,), dtype=torch.int64)

        return torch.as_tensor(img, dtype=torch.float32), target, elem.res, elem.image

class CardiacBatch(BatchDataset):

    def __init__(self, datasetfile, split=['base'], iterations=None, batch_size=None, res=None, seed=None):
        super(BatchDataset, self).__init__()
        self.init(datasetfile, split, iterations, batch_size, res, seed)
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
        img = sitk.ReadImage(elem.filepath)
        img = sitk.GetArrayFromImage(img)[elem.t, elem.slice, :, :]
        img = mut.norm01(img)

        mask = sitk.ReadImage(elem.filepath[:-7] + '_gt.nii.gz')
        mask = sitk.GetArrayFromImage(mask)[elem.t, elem.slice, :, :]

        if img.shape != self.outsize:
            img = self.crop_center_or_pad(img, self.outsize[0], self.outsize[1])
            mask = self.crop_center_or_pad(mask, self.outsize[0], self.outsize[1])

        return img[None, :, :], mask

    def __getitem__(self, index):
        elem = self.df.iloc[index]
        img, mask = self.load_image(elem)
        return torch.as_tensor(img, dtype=torch.float32), torch.as_tensor(mask,
                                                                          dtype=torch.long), elem.scanner, elem.filepath