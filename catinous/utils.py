import pytorch_lightning.loggers as pllogging
from py_jotools import mut
import argparse
import pandas as pd

LOGGING_FOLDER = '/project/catinous/tensorboard_logs/'
TRAINED_MODELS_FOLDER = '/project/catinous/trained_models/'
TRAINED_CACHE_FOLDER = '/project/catinous/trained_cache/'


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


def get_expname(hparams):
    if type(hparams) is argparse.Namespace:
        hparams = vars(hparams)
    hashed_params = mut.hash(hparams, length=10)
    expname = ''
    expname += 'cont' if hparams['continous'] else 'batch'
    expname += '_' + hparams['datasetfile'].split('_')[1]
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





