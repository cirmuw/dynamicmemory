from glob import glob
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
import argparse

def norm01(x):
    """Normalizes values in x to be between 0 and 255"""
    r = (x - np.min(x))
    m = np.max(r)
    if m > 0:
        r = np.divide(r, np.max(r))
    return r

def prepare_data(datasetpath, outputpath, seed=516165):
    df_dsinfo = pd.read_csv(f'{datasetpath}/201014_M&Ms_Dataset_Information_-_opendataset.csv')
    df_dsinfo['labels'] = True

    for p in glob(f'{datasetpath}/Training/Unlabeled/*'):
        df_dsinfo.loc[df_dsinfo['External code'] == p.split('/')[-1], 'labels'] = False

    df_dsinfo = df_dsinfo.loc[df_dsinfo.labels]
    df_base = df_dsinfo.loc[df_dsinfo.VendorName == 'Siemens']
    df_base = df_base.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_base['split'] = 'None'
    df_base.iloc[0:50].split = 'base'
    df_base.iloc[50:75].split = 'train'
    df_base.iloc[75:85].split = 'val'
    df_base.iloc[85:].split = 'test'

    df_phil = df_dsinfo.loc[df_dsinfo.VendorName == 'Philips']
    df_phil = df_phil.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_phil['split'] = 'None'
    df_phil.iloc[0:105].split = 'train'
    df_phil.iloc[105:115].split = 'val'
    df_phil.iloc[115:].split = 'test'

    df_ge = df_dsinfo.loc[df_dsinfo.VendorName == 'GE']
    df_ge = df_ge.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_ge['split'] = 'None'
    df_ge.iloc[0:30].split = 'train'
    df_ge.iloc[30:40].split = 'val'
    df_ge.iloc[40:].split = 'test'

    df_canon = df_dsinfo.loc[df_dsinfo.VendorName == 'Canon']
    df_canon = df_canon.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_canon['split'] = 'None'
    df_canon.iloc[0:30].split = 'train'
    df_canon.iloc[30:40].split = 'val'
    df_canon.iloc[40:].split = 'test'

    df_train = pd.concat([df_base, df_phil, df_ge, df_canon])
    df_train.groupby(['VendorName', 'split']).count()

    scanners = []
    imgs = []
    ts = []
    slices = []
    splits = []
    slicepath = []
    for i, row in df_train.iterrows():
        excode = row['External code']

        if os.path.exists(f'{datasetpath}/Training/Labeled/{excode}/'):
            filepath = f'{datasetpath}/Training/Labeled/{excode}/{excode}_sa.nii.gz'
            maskpath = f'{datasetpath}/Training/Labeled/{excode}/{excode}_sa_gt.nii.gz'
        elif os.path.exists(f'{datasetpath}/Testing/{excode}/'):
            filepath = f'{datasetpath}/Testing/{excode}/{excode}_sa.nii.gz'
            maskpath = f'{datasetpath}/Testing/{excode}/{excode}_sa_gt.nii.gz'
        elif os.path.exists(f'{datasetpath}/Validation/{excode}/'):
            filepath = f'{datasetpath}/Validation/{excode}/{excode}_sa.nii.gz'
            maskpath = f'{datasetpath}/Validation/{excode}/{excode}_sa_gt.nii.gz'


        img = sitk.ReadImage(filepath)
        imgd = sitk.GetArrayFromImage(img)
        mask = sitk.ReadImage(maskpath)
        maskd = sitk.GetArrayFromImage(mask)
        sl = img.GetSize()[2]

        slices.extend(list(range(0, sl)))
        imgs.extend([filepath] * sl)
        ts.extend([row.ED] * sl)
        scanners.extend([row.VendorName] * sl)
        splits.extend([row.split] * sl)

        for i in range(0, sl):
            img_sl = imgd[row.ED, i, :, :]
            img_sl = norm01(img_sl)
            os.makedirs(f'{outputpath}/{excode}/', exist_ok=True)
            np.save(f'{outputpath}/{excode}/{row.ED}_{i}.npy', img_sl)
            slicepath.append(f'{outputpath}/{excode}/{row.ED}_{i}.npy')
            mask_sl = maskd[row.ED, i, :, :]
            np.save(f'{outputpath}/{excode}/{row.ED}_{i}_gt.npy', mask_sl)

        slices.extend(list(range(0, sl)))
        imgs.extend([filepath] * sl)
        ts.extend([row.ES] * sl)
        scanners.extend([row.VendorName] * sl)
        splits.extend([row.split] * sl)

        for i in range(0, sl):
            img_sl = imgd[row.ED, i, :, :]
            img_sl = norm01(img_sl)
            np.save(f'{outputpath}/{excode}/{row.ES}_{i}.npy', img_sl)
            slicepath.append(f'{outputpath}/{excode}/{row.ED}_{i}.npy')
            mask_sl = maskd[row.ED, i, :, :]
            np.save(f'{outputpath}/{excode}/{row.ES}_{i}_gt.npy', mask_sl)

    df_train_slices = pd.DataFrame({'scanner': scanners, 'filepath': imgs, 'slicepath': slicepath, 't': ts, 'slice': slices, 'split': splits})
    df_train_slices.to_csv(f'{outputpath}/cardiacdatasetsplit.csv', index=False)

if __name__ == "__main__":
    # execute only if run as a script
    parser = argparse.ArgumentParser(description='Prepare the cardiac segmentation dataset for dynamic memory.')
    parser.add_argument('datasetpath', type=str, help='path to the downloaded M&Ms Challenge dataset')
    parser.add_argument('outputpath', type=str, help='path to the directory where the prepared dataframe and single slices are stored')
    args = parser.parse_args()

    prepare_data(os.path.abspath(args.datasetpath), os.path.abspath(args.outputpath))