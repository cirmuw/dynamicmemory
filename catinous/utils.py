import pytorch_lightning.loggers as pllogging
from py_jotools import mut
import argparse
import pandas as pd
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import os


LOGGING_FOLDER = '/project/catinous/tensorboard_logs/'
TRAINED_MODELS_FOLDER = '/project/catinous/trained_models/'
TRAINED_CACHE_FOLDER = '/project/catinous/trained_cache/'
RESPATH = '/project/catinous/results/'


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

    ##### hack hack hack hack, so don't have to recalculate results for cases without the EWC parameter
    if not hparams['EWC']:
        hparams.pop('EWC')
        hparams.pop('EWC_dataset')
        hparams.pop('EWC_lambda')
        hparams.pop('EWC_bn_off')

    if hparams['gram_weights'] == [1, 1, 1, 1]:
        hparams.pop('gram_weights')

    hashed_params = mut.hash(hparams, length=10)
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

    ##### hack hack hack hack, so don't have to recalculate results for cases without the EWC parameter
    if not hparams['EWC']:
        hparams.pop('EWC')
        hparams.pop('EWC_dataset')
        hparams.pop('EWC_lambda')
        hparams.pop('EWC_bn_off')

    if hparams['gram_weights'] == [1, 1, 1, 1]:
        hparams.pop('gram_weights')

    hashed_params = mut.hash(hparams, length=10)
    expname = ''
    expname += 'cont' if hparams['continous'] else 'batch'
    expname += '_' + hparams['datasetfile'].split('_')[1]
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


def save_cache_to_csv(cache, savepath):
    df_cache = pd.DataFrame({'filepath':[ci.filepath for ci in cache], 'label': [ci.label.cpu().numpy()[0] for ci in cache], 'res': [ci.res for ci in cache], 'traincounter': [ci.traincounter for ci in cache]})
    df_cache.to_csv(savepath, index=False, index_label=False)


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