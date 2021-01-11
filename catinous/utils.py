import pytorch_lightning.loggers as pllogging
from pytorch_lightning.utilities.parsing import AttributeDict
import argparse
import pandas as pd
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import os
import hashlib
import pickle
import skimage.transform
import numpy as np
import torchvision.models.segmentation
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pydicom as pyd

LOGGING_FOLDER = '/project/catinous/tensorboard_logs/'
TRAINED_MODELS_FOLDER = '/project/catinous/trained_models/'
TRAINED_MEMORY_FOLDER = '/project/catinous/trained_cache/'
RESPATH = '/project/catinous/results/'

def sort_dict(input_dict):
    dict_out = {}
    keys = list(input_dict.keys())
    keys.sort()
    for key in keys:
        if type(input_dict[key]) is dict:
            value = sort_dict(input_dict[key])
        else:
            value = input_dict[key]
        dict_out[key] = value
    return dict_out

def hash(item, length=40):
    assert (type(item) is dict)
    item = sort_dict(item)
    return hashlib.sha1(pickle.dumps(item)).hexdigest()[0:length]

def resize(img, size, order=1, anti_aliasing=True):
    for i in range(len(size)):
        if size[i] is None:
            size[i] = img.shape[i]
    return skimage.transform.resize(img, size, order=order, mode='reflect', anti_aliasing=anti_aliasing, preserve_range=True)

def norm01(x):
    """Normalizes values in x to be between 0 and 255"""
    r = (x - np.min(x))
    m = np.max(r)
    if m > 0:
        r = np.divide(r, np.max(r))
    return r

def default_params(dparams, params):
    """Copies all key value pairs from params to dparams if not present"""
    matched_params = dparams.copy()
    default_keys = dparams.keys()
    param_keys = params.keys()
    for key in param_keys:
        matched_params[key] = params[key]
        if key in default_keys:
            if (type(params[key]) is dict) and (type(dparams[key]) is dict):
                matched_params[key] = default_params(dparams[key], params[key])
    return matched_params


def pllogger(hparams):
    return pllogging.TestTubeLogger(LOGGING_FOLDER, name=get_expname(hparams))


def get_expname_age(hparams):
    if type(hparams) is argparse.Namespace:
        hparams = vars(hparams).copy()
    elif type(hparams) is AttributeDict:
        hparams = dict(hparams)

    ##### hack hack hack hack, so don't have to recalculate results for cases without the EWC parameter
    if not hparams['EWC']:
        hparams.pop('EWC')
        hparams.pop('EWC_dataset')
        hparams.pop('EWC_lambda')
        hparams.pop('EWC_bn_off')

    if hparams['gram_weights'] == [1, 1, 1, 1]:
        hparams.pop('gram_weights')

    hashed_params = hash(hparams, length=10)
    expname = ''
    expname += 'cont' if hparams['continous'] else 'batch'
    expname += '_' + os.path.basename(hparams['datasetfile'])[:-4]
    if 'logits' in hparams.keys(): #Hackydyhack
        expname += '_logits'
    if hparams['base_model']:
        expname += '_basemodel_' + hparams['base_model'].split('_')[1]
    if hparams['continous']:
        expname += '_fmiss' if hparams['force_misclassified'] else ''
        expname += '_cache' if hparams['use_cache'] else '_nocache'
        expname += '_tf{}'.format(str(hparams['transition_phase_after']).replace('.', ''))
    else:
        expname += '_' + '-'.join(hparams['noncontinous_train_splits'])
    expname += '_'+str(hparams['run_postfix'])
    expname += '_'+hashed_params
    return expname

def get_expname(hparams):
    if type(hparams) is argparse.Namespace:
        hparams = vars(hparams).copy()
    elif type(hparams) is AttributeDict:
        hparams = dict(hparams)

    ##### hack hack hack hack, so don't have to recalculate results for cases without the EWC parameter
    if 'EWC' in hparams:
        if not hparams['EWC']:
            hparams.pop('EWC')
            hparams.pop('EWC_dataset')
            hparams.pop('EWC_lambda')
            hparams.pop('EWC_bn_off')

    if 'gram_weights' in hparams:
        if hparams['gram_weights'] == [1, 1, 1, 1]:
            hparams.pop('gram_weights')

    hashed_params = hash(hparams, length=10)
    expname = ''
    expname += hparams['task']
    expname += '_cont' if hparams['continuous'] else '_batch'
    expname += '_' + os.path.basename(hparams['datasetfile'])[:-4]
    if 'logits' in hparams.keys(): #Hackydyhack
        expname += '_logits'
    if hparams['base_model']:
        expname += '_basemodel_' + hparams['base_model'].split('_')[1]
    if hparams['continuous']:
        expname += '_mem' if hparams['use_memory'] else '_nomem'
        expname += '_tf{}'.format(str(hparams['transition_phase_after']).replace('.', ''))
    else:
        expname += '_' + '-'.join(hparams['noncontinuous_train_splits'])
    expname += '_'+str(hparams['run_postfix'])
    expname += '_'+hashed_params
    return expname


def save_cache_to_csv(cache, savepath):
    df_cache = pd.DataFrame({'filepath':[ci.filepath for ci in cache], 'scanner': [ci.scanner for ci in cache], 'traincounter': [ci.traincounter for ci in cache]})
    df_cache.to_csv(savepath, index=False, index_label=False)

def gram_matrix(input):
    # taken from: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    grams = []

    for i in range(a):
        features = input[i].view(b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        grams.append(G.div(b * c * d))

    return grams

def gram_matrix_3d(input):
    # taken from: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    a, b, c, d, e = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    grams = []

    for i in range(a):
        features = input[i].view(b, c * d * e)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        grams.append(G.div(b * c * d * e))

    return grams

def load_model(modelstr: str):
    if modelstr == 'resnet':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential(
            *[nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.Linear(512, 1)])

        gramlayers = [model.layer1[-1].conv1,
                      model.layer2[-1].conv1,
                      model.layer3[-1].conv1,
                     model.layer4[-1].conv1]
    elif modelstr == 'fcn':
        model = models.segmentation.fcn_resnet50(num_classes=4, progress=False)
        model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        gramlayers = [
            model.backbone.layer1[-1].conv1,
            model.backbone.layer2[-1].conv1,
            model.backbone.layer3[-1].conv1,
            model.backbone.layer4[-1].conv1
        ]
    elif modelstr == 'rnn':
        num_classes = 3  # 0=background, 1=begnin, 2=malignant
        # load a model pre-trained pre-trained on COCO
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        gramlayers = [
            model.backbone.body.layer1[-1].conv1,
            model.backbone.body.layer2[-1].conv1,
            model.backbone.body.layer3[-1].conv1,
            model.backbone.body.layer4[-1].conv1
        ]
    else:
        raise NotImplementedError(f'model {modelstr} not implemented')

    return model, gramlayers


def load_model_stylemodel(modelstr: str):
    stylemodel = models.resnet50(pretrained=True)

    if modelstr == 'resnet':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential(
            *[nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.Linear(512, 1)])
    elif modelstr == 'fcn':
        model = models.segmentation.fcn_resnet50(num_classes=4)
        model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        stylemodel.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    elif modelstr == 'rnn':
        num_classes = 3  # 0=background, 1=begnin, 2=malignant
        # load a model pre-trained pre-trained on COCO
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        raise NotImplementedError(f'model {modelstr} not implemented')

    gramlayers = [stylemodel.layer1[-1].conv1,
                  stylemodel.layer2[-1].conv1,
                  stylemodel.layer3[-1].conv1,
                  stylemodel.layer4[-1].conv1]
    stylemodel.eval()

    return model, stylemodel, gramlayers

# from https://github.com/moskomule/ewc.pytorch/
class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            input = variable(input[0].float())
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        print('ewcloss: %f' % loss.data.cpu().numpy())
        return loss


class BCEWithLogitWithEWCLoss(torch.nn.BCEWithLogitsLoss):

    def __init__(self, lambda_p):
        super(BCEWithLogitWithEWCLoss, self).__init__()
        self.register_buffer('lambda_p',lambda_p)

    def forward(self, input, target, penalty):
        bce = F.binary_cross_entropy_with_logits(input, target,
                                           self.weight,
                                           pos_weight=self.pos_weight,
                                           reduction=self.reduction)
        return bce + self.lambda_p * penalty


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)



def collate_fn(batch):
    return tuple(zip(*batch))

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def filter_boxes_area(boxes, scores, min_area=10):
    out_boxes = []
    out_scores = []
    for i, b in enumerate(boxes):
        area = (b[3] - b[1]) * (b[2] - b[0])
        if area > min_area:
            out_boxes.append(b)
            out_scores.append(scores[i])

    return np.array(out_boxes), np.array(out_scores)


def correct_boxes(boxes_np, scores_np, x_shift=112, y_shift=112):
    #if len(boxes_np[0]) > 0:
    #    boxes_np[0][:, 0] += x_shift
    #    boxes_np[0][:, 1] += y_shift
    #    boxes_np[0][:, 2] += x_shift
    #    boxes_np[0][:, 3] += y_shift

    #if len(boxes_np[2]) > 0:
    #    boxes_np[2][:, 1] += y_shift * 2
    #    boxes_np[2][:, 3] += y_shift * 2

    #if len(boxes_np[3]) > 0:
    #    boxes_np[3][:, 0] += x_shift * 2
    #    boxes_np[3][:, 2] += x_shift * 2

    #if len(boxes_np[4]) > 0:
    #    boxes_np[4][:, 0] += x_shift * 2
    #    boxes_np[4][:, 2] += x_shift * 2
    #    boxes_np[4][:, 1] += y_shift * 2
    #    boxes_np[4][:, 3] += y_shift * 2

    # there is a better way for sure... move fast and break things
    #final_boxes = []
    #final_boxes.extend(boxes_np[0])
    #final_boxes.extend(boxes_np[1])
    #final_boxes.extend(boxes_np[2])
    #final_boxes.extend(boxes_np[3])
    #final_boxes.extend(boxes_np[4])

    #final_scores = []
    #final_scores.extend(scores_np[0])
    #final_scores.extend(scores_np[1])
    #final_scores.extend(scores_np[2])
    #final_scores.extend(scores_np[3])
    #final_scores.extend(scores_np[4])

    #if len(final_boxes) > 0:
    if len(boxes_np) > 0:
        bidx = torch.ops.torchvision.nms(torch.as_tensor(boxes_np), torch.as_tensor(scores_np), 0.2)

        if len(bidx) == 1:
            final_scores = [np.array(scores_np)[bidx]]
            final_boxes = [np.array(boxes_np)[bidx]]
        else:
            final_scores = np.array(scores_np)[bidx]
            final_boxes = np.array(boxes_np)[bidx]

        return final_boxes, final_scores
    else:
        return boxes_np, scores_np


def load_box_annotation(elem, cropped_to=None, shiftx_aug=0, shifty_aug=0, validation=False):
    dcm = pyd.read_file(elem.image)
    x = elem.coordX
    y = elem.coordY
    diameter = elem.diameter_mm
    spacing = float(dcm.PixelSpacing[0])

    if not validation:
        if cropped_to is not None:
            x -= (dcm.Rows - cropped_to[0]) / 2
            y -= (dcm.Columns - cropped_to[1]) / 2
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