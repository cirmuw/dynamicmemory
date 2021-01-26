from catinous.dataset.BatchDataset import CardiacBatch, LIDCBatch
import catinous.dynamicmemory.DynamicMemoryModel as dmodel
import torch
from torch.utils.data import DataLoader
from py_jotools import augmentation, mut
import pandas as pd
import numpy as np
import catinous.utils as cutils

def eval_cardiac(hparams, outfile):
    device = torch.device('cuda')

    dl_test = DataLoader(CardiacBatch(hparams['datasetfile'], split=['test']), batch_size=16)
    print('dl loaded')

    model, _, _, _ = dmodel.trained_model(hparams, training=False)
    print('reading model done')
    model.to(device)
    model.eval()

    scanners = []
    dice_1 = []
    dice_2 = []
    dice_3 = []
    shifts = []
    img = []

    print('starting to eval on model')
    for batch in dl_test:
        x, y, scanner, filepath = batch
        x = x.to(device)
        y_hat = model.forward(x)['out']
        y_hat_flat = torch.argmax(y_hat, dim=1).detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        for i, m in enumerate(y):
            scanners.append(scanner[i])
            dice_1.append(mut.dice(y[i], y_hat_flat[i], classi=1))
            dice_2.append(mut.dice(y[i], y_hat_flat[i], classi=2))
            dice_3.append(mut.dice(y[i], y_hat_flat[i], classi=3))
            img.append(filepath[i])
            shifts.append('None')
    print('finished eval on model')

    modelpath = dmodel.cached_path(hparams)

    for s in ['Canon', 'GE', 'Philips']:
        shiftmodelpath = f'{modelpath[:-3]}_shift_{s}.pt'
        print('starting load shift model', s, shiftmodelpath)
        model.model.load_state_dict(torch.load(shiftmodelpath, map_location=device))
        model.freeze()
        print('starting to eval on shiftmodel', s)

        for batch in dl_test:
            x, y, scanner, filepath = batch
            x = x.to(device)
            y_hat = model.forward(x)['out']
            y_hat_flat = torch.argmax(y_hat, dim=1).detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            for i, m in enumerate(y):
                scanners.append(scanner[i])
                dice_1.append(mut.dice(y[i], y_hat_flat[i], classi=1))
                dice_2.append(mut.dice(y[i], y_hat_flat[i], classi=2))
                dice_3.append(mut.dice(y[i], y_hat_flat[i], classi=3))
                img.append(filepath[i])

                shifts.append(s)
        print('finished to eval on shiftmodel', s)

    df_results = pd.DataFrame({'scanner': scanners, 'dice_1': dice_1, 'dice_2': dice_2, 'dice_3': dice_3, 'shift': shifts})
    df_results.to_csv(outfile, index=False)


def eval_cardiac_batch(hparams, outfile):
    device = torch.device('cuda')

    dl_test = DataLoader(CardiacBatch(hparams['datasetfile'], split=['test']), batch_size=16)
    print('dl loaded')

    model, _, _, _ = dmodel.trained_model(hparams, training=False)
    print('reading model done')
    model.to(device)
    model.eval()

    scanners = []
    dice_1 = []
    dice_2 = []
    dice_3 = []
    img = []

    print('starting to eval on model')
    for batch in dl_test:
        x, y, scanner, filepath = batch
        x = x.to(device)
        y_hat = model.forward(x)['out']
        y_hat_flat = torch.argmax(y_hat, dim=1).detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        for i, m in enumerate(y):
            scanners.append(scanner[i])
            dice_1.append(mut.dice(y[i], y_hat_flat[i], classi=1))
            dice_2.append(mut.dice(y[i], y_hat_flat[i], classi=2))
            dice_3.append(mut.dice(y[i], y_hat_flat[i], classi=3))
            img.append(filepath[i])
    print('finished eval on model')

    df_results = pd.DataFrame({'scanner': scanners, 'dice_1': dice_1, 'dice_2': dice_2, 'dice_3': dice_3})
    df_results.to_csv(outfile, index=False)


def ap_model_hparams(hparams, split='test'):
    device = torch.device('cuda')
    model, logs, df_mem, expname = dmodel.trained_model(hparams, training=False)
    model.to(device)
    model.eval()
    recalls, precision = ap_model(model, split)
    return recalls, precision, model


def ap_model(model, split='test'):
    recalls = {'ges': [], 'geb': [], 'sie': [], 'time_siemens': []}
    precision = {'ges': [], 'geb': [], 'sie': [], 'time_siemens': []}
    device = torch.device('cuda')

    for res in ['ges', 'geb', 'sie', 'time_siemens']:
        ds_test = LIDCBatch('/project/catinous/lungnodulesfinalpatientsplit.csv',
                            cropped_to=(288, 288), split=split, res=res, validation=True)

        iou_thres = 0.2

        overall_true_pos = dict()
        overall_false_pos = dict()
        overall_false_neg = dict()
        overall_boxes_count = dict()
        for k in np.arange(0.0, 1.01, 0.05):
            overall_true_pos[k] = 0
            overall_false_pos[k] = 0
            overall_false_neg[k] = 0
            overall_boxes_count[k] = 0

        for batch in ds_test:
            img_batch, annot, res, image = batch
            img_batch = img_batch[None, :, :, :]
            img_batch = img_batch.to(device)

            out = model.model(img_batch)
            out_boxes = [cutils.filter_boxes_area(out[i]['boxes'].cpu().detach().numpy(),
                                                  out[i]['scores'].cpu().detach().numpy()) for i in range(len(out))]
            boxes_np = [b[0] for b in out_boxes]
            scores_np = [b[1] for b in out_boxes]

            final_boxes, final_scores = cutils.correct_boxes(boxes_np[0], scores_np[0])

            gt = annot['boxes']
            #if res=='time_siemens':
            #    print('new time series')
            #    print(gt)
            #    print(final_boxes, len(final_boxes))
            for k in np.arange(0.0, 1.01, 0.05):
                false_positives = 0
                false_negatives = 0
                true_positives = 0
                detected = [False]*len(gt)
                boxes_count = 0
                if len(final_boxes) > 0:
                    for i, b in enumerate(final_boxes):
                        if final_scores[i] > k:
                            boxes_count += 1
                            detected_gt = False
                            for j, g in enumerate(gt):
                                #if res=='time_siemens':
                                    #print(cutils.bb_intersection_over_union(g, b), 'intersect')
                                if cutils.bb_intersection_over_union(g, b) > iou_thres:
                                    detected[j] = True
                                    detected_gt = True
                            if not detected_gt:
                                false_positives += 1
                #if res == 'time_siemens':
                #    print(detected)
                for d in detected:
                    if d:
                        true_positives+=1
                    else:
                        false_negatives+=1

                    #if detected:
                    #    true_positives += 1
                    #else:
                    #    false_negatives += 1
                overall_true_pos[k] += true_positives
                overall_false_pos[k] += false_positives
                overall_false_neg[k] += false_negatives
                overall_boxes_count[k] += boxes_count
        #if res=='time_siemens':
            #print(overall_boxes_count, overall_true_pos, overall_false_pos, overall_false_neg)
        for k in np.arange(0.0, 1.01, 0.05):
            if (overall_false_neg[k] + overall_true_pos[k]) == 0:
                recalls[res].append(0.0)
            else:
                recalls[res].append(overall_true_pos[k] / (overall_false_neg[k] + overall_true_pos[k]))
            if (overall_false_pos[k] + overall_true_pos[k]) == 0:
                precision[res].append(0.0)
            else:
                precision[res].append(overall_true_pos[k] / (overall_false_pos[k] + overall_true_pos[k]))
    return recalls, precision

def recall_precision_to_ap(recalls, precisions):
    aps = dict()
    for res in ['ges', 'geb', 'sie', 'time_siemens']:
        prec = np.array(precisions[res])
        rec = np.array(recalls[res])
        ap = []
        for t in np.arange(0.0, 1.01, 0.1):
            prec_arr = prec[rec > t]
            if len(prec_arr) == 0:
                ap.append(0.0)
            else:
                ap.append(prec_arr.max())
        aps[res] = np.array(ap).mean()
    return aps

def get_ap_for_res(hparams, split='test', shifts=None):
    device = torch.device('cuda')
    recalls, precisions, model = ap_model_hparams(hparams, split)
    aps = recall_precision_to_ap(recalls, precisions)
    df_aps = pd.DataFrame([aps])

    if shifts is not None:
        df_aps['shift'] = 'None'

        modelpath = dmodel.cached_path(hparams)

        for s in shifts:
            shiftmodelpath = f'{modelpath[:-3]}_shift_{s}.pt'
            print('starting load shift model', s, shiftmodelpath)
            model.model.load_state_dict(torch.load(shiftmodelpath, map_location=device))
            model.freeze()
            print('starting to eval on shiftmodel', s)

            recalls, precisions = ap_model(model, split)
            aps = recall_precision_to_ap(recalls, precisions)
            aps = pd.DataFrame([aps])
            aps['shift'] = s
            df_aps = df_aps.append(aps)
    return df_aps

def eval_lidc_cont(hparams, seeds=None, split='test', shifts=None):
    outputfile = f'/project/catinous/results/lidc/{cutils.get_expname(hparams)}_meanaverageprecision.csv'
    seeds_aps = pd.DataFrame()

    if seeds is not None:
        for i, seed in enumerate(seeds):
            hparams['seed'] = seed
            hparams['run_postfix'] = i+1
            aps = get_ap_for_res(hparams, split=split, shifts=shifts)
            aps['seed'] = seed
            seeds_aps = seeds_aps.append(aps)
    else:
        aps = get_ap_for_res(hparams, split=split, shifts=shifts)
        seeds_aps = seeds_aps.append(aps)

    seeds_aps.to_csv(outputfile, index=False)