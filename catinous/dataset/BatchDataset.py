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
            if type(res) is list:
                selection = np.any([self.df.scanner == x for x in res], axis=0)
            else:
                selection = self.df.scanner == res

            self.df = self.df.loc[selection]
            self.df = self.df.reset_index()

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
                 cropped_to=(288, 288), validation=False):
        super(BatchDataset, self).__init__()
        self.init(datasetfile, split, iterations, batch_size, res, seed)

        self.cropped_to = cropped_to
        self.validation = validation

        #if validation:
        #    self.df = self.df.sort_values('patient_id').reset_index(drop=True)

        self.df_multiplenodules = pd.read_csv('/project/catinous/lungnodules_allnodules.csv')


    def load_image(self, path, shiftx_aug=0, shifty_aug=0):
        #try:
        #    img = pyd.read_file(path).pixel_array
        #except Exception as e:
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

    def load_image_validation(self, elem):
        #dcm = pyd.read_file(elem.image)
        #img = dcm.pixel_array
        dcm = sitk.ReadImage(elem.image)
        img = sitk.GetArrayFromImage(dcm)

        x = elem.x1
        y = elem.y1
        x2 = elem.x2
        y2 = elem.y2

        if self.cropped_to is not None:
            w = img.shape[1]
            h = img.shape[2]
            x_shift = int((w - self.cropped_to[0]) / 2)
            y_shift = int((h - self.cropped_to[1]) / 2)
            s1 = x_shift
            e1 = int(s1 + self.cropped_to[0])
            s2 = y_shift
            e2 = int(s2 + self.cropped_to[1])

            #x -= (dcm.Rows - self.cropped_to[0]) / 2
            #y -= (dcm.Columns - self.cropped_to[1]) / 2
            #x2 -= (dcm.Rows - self.cropped_to[0]) / 2
            #y2 -= (dcm.Columns - self.cropped_to[1]) / 2

            x -= s2
            y -= s1
            x2 -= s2
            y2 -= s1


            if x<0:
                s2 -= (x*-1) + 5
                e2 -= (x*-1) + 5

                x2 = 5+(x2-x)
                x = 5
            elif x>self.cropped_to[1]:
                s2 += (x2-self.cropped_to[1]+5)
                e2 += (x2-self.cropped_to[1]+5)
                if e2>dcm.GetSize()[0]:
                    s2 -= (e2-dcm.GetSize()[0])
                    e2 = min(e2,dcm.GetSize()[0])

                x = self.cropped_to[1] - (x2-x) - 5
                x2 = self.cropped_to[1]-5

            if y<0:
                s1 += (y * -1) + 5
                e1 += (y * -1) + 5

                y2 = 5 + (y2 - y)
                y = 5



            im_crop = img[:, int(s1):int(e1), int(s2):int(e2)]
            im_crop = mut.intensity_window(im_crop, low=-1024, high=1500)
            try:
                im_crop = mut.norm01(im_crop)
                pass
            except Exception as e:
                print(im_crop.shape, s1, e1, s2, e2, y, y2, elem.image)
                raise e
        else:
            im_crop = img
            im_crop = mut.intensity_window(im_crop, low=-1024, high=1500)
            im_crop = mut.norm01(im_crop)

        xs = []
        x2s = []
        ys = []
        y2s = []
        for i, row in self.df_multiplenodules.loc[self.df_multiplenodules.image==elem.image].iterrows():
            print('multiple')
            x1_new = row.x1-s2
            x2_new = row.x2-s2

            y1_new = row.y1-s1
            y2_new = row.y2-s1

            if x1_new>0 and x1_new<self.cropped_to[0] and y1_new>0 and y1_new<self.cropped_to[1]:
                xs.append(x1_new)
                x2s.append(x2_new)

                ys.append(y1_new)
                y2s.append(y2_new)

        if xs==[]:
            box = np.zeros((1, 4))
            box[0, 0] = x
            box[0, 1] = y
            box[0, 2] = x2
            box[0, 3] = y2
        else:
            box = np.zeros((len(xs), 4))
            #box[0, 0] = x
            #box[0, 1] = y
            #box[0, 2] = x2
            #box[0, 3] = y2

            for j, x in enumerate(xs):
                box[j, 0] = x
                box[j, 1] = ys[j]
                box[j, 2] = x2s[j]
                box[j, 3] = y2s[j]

        return np.tile(im_crop, [3, 1, 1]), box

    def load_annotation(self, elem, shiftx_aug=0, shifty_aug=0, ):
        #dcm = pyd.read_file(elem.image, force=True)
        dcm = sitk.ReadImage(elem.image)

        x = elem.x1
        y = elem.y1
        x2 = elem.x2
        y2 = elem.y2


        if self.cropped_to is not None:
            x -= (dcm.GetSize()[0] - self.cropped_to[0]) / 2
            y -= (dcm.GetSize()[1] - self.cropped_to[1]) / 2
            x2 -= (dcm.GetSize()[0] - self.cropped_to[0]) / 2
            y2 -= (dcm.GetSize()[1] - self.cropped_to[1]) / 2

        y -= shiftx_aug
        x -= shifty_aug
        y2 -= shiftx_aug
        x2 -= shifty_aug


        xs = []
        x2s = []
        ys = []
        y2s = []
        for i, row in self.df_multiplenodules.loc[self.df_multiplenodules.image == elem.image].iterrows():

            x1_new =  row.x1 - shifty_aug
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

        if not self.validation:
            if self.cropped_to is None:
                shiftx_aug = 0
                shifty_aug = 0
            else:
                shiftx_aug = random.randint(-100, 100)
                shifty_aug = random.randint(-100, 100)

            img = self.load_image(elem.image, shiftx_aug, shifty_aug)
            annotation = self.load_annotation(elem, shiftx_aug, shifty_aug)
        else:
            img, annotation = self.load_image_validation(elem)

        target = {}
        target['boxes'] = torch.as_tensor(annotation, dtype=torch.float32)
        target['labels'] = torch.as_tensor([elem.bin_malignancy + 1]*len(annotation), dtype=torch.int64)
        target['image_id'] = torch.tensor([index]*len(annotation))

        target['area'] = torch.as_tensor(
            ((annotation[:, 3] - annotation[:, 1]) * (annotation[:, 2] - annotation[:, 0])))
        target['iscrowd'] = torch.zeros((len(annotation)), dtype=torch.int64)

        return torch.as_tensor(img, dtype=torch.float32), target, elem.scanner, elem.image
        #return img, target, elem.scanner, elem.image
        
class CardiacBatch(BatchDataset):

    def __init__(self, datasetfile, split=['base'], iterations=None, batch_size=None, res=None, seed=None):
        super(BatchDataset, self).__init__()
        self.init(datasetfile, split, iterations, batch_size, res, seed)
        self.outsize = (240, 196)

        print('init cardiac batch with datasetfile', datasetfile)


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
        # img = sitk.ReadImage(elem.filepath)
        # img = sitk.GetArrayFromImage(img)[elem.t, elem.slice, :, :]
        # img = mut.norm01(img)

        # mask = sitk.ReadImage(elem.filepath[:-7] + '_gt.nii.gz')
        # mask = sitk.GetArrayFromImage(mask)[elem.t, elem.slice, :, :]

        # if img.shape != self.outsize:
        #    img = self.crop_center_or_pad(img, self.outsize[0], self.outsize[1])
        #    mask = self.crop_center_or_pad(mask, self.outsize[0], self.outsize[1])

        img = np.load(elem.slicepath)
        mask = np.load(elem.slicepath[:-4] + '_gt.npy')

        return img[None, :, :], mask

    def __getitem__(self, index):
        elem = self.df.iloc[index]
        img, mask = self.load_image(elem)
        return torch.as_tensor(img, dtype=torch.float32), torch.as_tensor(mask,
                                                                          dtype=torch.long), elem.scanner, elem.filepath