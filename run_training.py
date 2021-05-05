import yaml
import dynamicmemory.DynamicMemoryModel as dmodel
import argparse

def train_paper():
    seeds = [1654130, 6654961, 5819225, 1215862, 132054] #random numbers generated once and fixed to ensure reproducibility

    # 1. Run base training as all other training runs need that information
    with open('training_configs/cardiac_base.yml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    model, logs, df_mem, exp_name_base = dmodel.trained_model(params['trainparams'], params['settings'])

    for i, seed in enumerate(seeds):
        with open('training_configs/cardiac_dynamicmemory.yml') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        params['trainparams']['basemodel'] = exp_name_base + '.pt'
        params['trainparams']['memorymaximum'] = 64
        params['trainparams']['run_postfix'] = i+1
        params['trainparams']['seed'] = seed
        dmodel.trained_model(params['trainparams'], params['settings'])

        params['trainparams']['memorymaximum'] = 128
        dmodel.trained_model(params['trainparams'], params['settings'])

        with open('training_configs/cardiac_dynamicmemory_pd.yml') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        params['trainparams']['basemodel'] = exp_name_base + '.pt'
        params['trainparams']['memorymaximum'] = 64
        params['trainparams']['run_postfix'] = i+1
        params['trainparams']['seed'] = seed
        dmodel.trained_model(params['trainparams'], params['settings'])

        params['trainparams']['memorymaximum'] = 128
        dmodel.trained_model(params['trainparams'], params['settings'])

        with open('training_configs/cardiac_naive.yml') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        params['trainparams']['basemodel'] = exp_name_base + '.pt'
        params['trainparams']['run_postfix'] = i + 1
        params['trainparams']['seed'] = seed
        dmodel.trained_model(params['trainparams'], params['settings'])

        with open('training_configs/cardiac_EWC.yml') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        params['trainparams']['basemodel'] = exp_name_base + '.pt'
        params['trainparams']['run_postfix'] = i + 1
        params['trainparams']['seed'] = seed
        dmodel.trained_model(params['trainparams'], params['settings'])

def train_config(configfile):
    with open(configfile) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    model, logs, df_mem, exp_name = dmodel.trained_model(params['trainparams'], params['settings'])

    print('successfully trained model', exp_name)


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

