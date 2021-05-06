import yaml
import dynamicmemory.DynamicMemoryModel as dmodel
import argparse
import utils as dmutils
import torch
import os
from pytorch_lightning import Trainer
from dynamicmemory.CardiacDynamicMemoryModel import CardiacDynamicMemoryModel
from dynamicmemory.LIDCDynamicMemoryModel import LIDCDynamicMemoryModel
import pandas as pd

def train_paper():
    seeds = [1654130, 6654961, 5819225, 1215862, 132054] #random numbers generated once and fixed to ensure reproducibility

    # 1. Run base training as all other training runs need that information
    with open('training_configs/cardiac_base.yml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    model, logs, df_mem, exp_name_base = trained_model(params['trainparams'], params['settings'])

    for i, seed in enumerate(seeds):
        with open('training_configs/cardiac_dynamicmemory.yml') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        params['trainparams']['basemodel'] = exp_name_base + '.pt'
        params['trainparams']['memorymaximum'] = 64
        params['trainparams']['run_postfix'] = i+1
        params['trainparams']['seed'] = seed
        dmodel.trained_model(params['trainparams'], params['settings'])

        params['trainparams']['memorymaximum'] = 128
        trained_model(params['trainparams'], params['settings'])

        with open('training_configs/cardiac_dynamicmemory_pd.yml') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        params['trainparams']['basemodel'] = exp_name_base + '.pt'
        params['trainparams']['memorymaximum'] = 64
        params['trainparams']['run_postfix'] = i+1
        params['trainparams']['seed'] = seed
        trained_model(params['trainparams'], params['settings'])

        params['trainparams']['memorymaximum'] = 128
        trained_model(params['trainparams'], params['settings'])

        with open('training_configs/cardiac_naive.yml') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        params['trainparams']['basemodel'] = exp_name_base + '.pt'
        params['trainparams']['run_postfix'] = i + 1
        params['trainparams']['seed'] = seed
        trained_model(params['trainparams'], params['settings'])

        with open('training_configs/cardiac_EWC.yml') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        params['trainparams']['basemodel'] = exp_name_base + '.pt'
        params['trainparams']['run_postfix'] = i + 1
        params['trainparams']['seed'] = seed
        trained_model(params['trainparams'], params['settings'])

def train_config(configfile):
    with open(configfile) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    model, logs, df_mem, exp_name = trained_model(params['trainparams'], params['settings'])

    print('successfully trained model', exp_name)


def trained_model(hparams, settings, training=True):
    df_memory = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    settings = argparse.Namespace(**settings)
    os.makedirs(settings.TRAINED_MODELS_DIR, exist_ok=True)
    os.makedirs(settings.TRAINED_MEMORY_DIR, exist_ok=True)
    os.makedirs(settings.RESULT_DIR, exist_ok=True)

    if hparams['task'] == 'cardiac':
        model = CardiacDynamicMemoryModel(hparams=hparams, modeldir=settings.TRAINED_MODELS_DIR, device=device, training=training)
    elif hparams['task'] == 'lidc':
        model = LIDCDynamicMemoryModel(hparams=hparams, modeldir=settings.TRAINED_MODELS_DIR, device=device, training=training)

    exp_name = dmutils.get_expname(hparams)
    print('expname', exp_name)
    weights_path = cached_path(hparams, settings.TRAINED_MODELS_DIR)

    if not os.path.exists(weights_path):
        if training:
            logger = dmutils.pllogger(hparams, settings.LOGGING_DIR)
            trainer = Trainer(gpus=1, max_epochs=1, logger=logger,
                              val_check_interval=model.hparams.val_check_interval,
                              checkpoint_callback=False, progress_bar_refresh_rate=0)
            trainer.fit(model)
            model.freeze()
            torch.save(model.state_dict(), weights_path)
            if model.hparams.continuous and model.hparams.use_memory:
                dmutils.save_cache_to_csv(model.trainingsmemory.memorylist,
                                          settings.TRAINED_MEMORY_DIR + exp_name + '.csv')
        else:
            model = None
    else:
        print('Read: ' + cached_path(hparams, settings.TRAINED_MODELS_DIR))
        model.load_state_dict(torch.load(cached_path(hparams, settings.TRAINED_MODELS_DIR), map_location=device))
        model.freeze()

    if model.hparams.continuous and model.hparams.use_memory:
        if os.path.exists(settings.TRAINED_MEMORY_DIR + exp_name + '.csv'):
            df_memory = pd.read_csv(settings.TRAINED_MEMORY_DIR + exp_name + '.csv')
        else:
            df_memory = None

    # always get the last version
    if os.path.exists(settings.LOGGING_DIR + exp_name):
        max_version = max([int(x.split('_')[1]) for x in os.listdir(settings.LOGGING_DIR + exp_name)])
        logs = pd.read_csv(settings.LOGGING_DIR + exp_name + '/version_{}/metrics.csv'.format(max_version))
    else:
        logs = None

    return model, logs, df_memory, exp_name + '.pt'


def is_cached(hparams, trained_dir):
    exp_name = dmutils.get_expname(hparams)
    return os.path.exists(trained_dir + exp_name + '.pt')


def cached_path(hparams, trained_dir):
    exp_name = dmutils.get_expname(hparams)
    return trained_dir + exp_name + '.pt'


if __name__ == "__main__":
    # execute only if run as a script
    parser = argparse.ArgumentParser(description='Run a training with the dynamic memory framework.')
    parser.add_argument('--config', type=str, help='path to a config file (yml) to run')
    parser.add_argument('-p',
                           '--paper',
                           action='store_true',
                           help='run all experiments reported in the paper')
    args = parser.parse_args()

    if args.paper:
        train_paper()
    elif args.config is not None:
        train_config(args.config)

