import pytorch_lightning.logging as pllogging
from py_jotools import mut

LOGGING_FOLDER = '/project/catinous/tensorboard_logs/'

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
    pllogging.TestTubeLogger(LOGGING_FOLDER, name=expname(hparams))

def expname(hparams):
    mut.hash(hparams, length=10)