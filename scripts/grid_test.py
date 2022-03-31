from copy import deepcopy
from commander import get_config, load_model, setup_trainer
import argparse
import numpy as np
import os
from data import get_test_dataset
from toolbox.planner import DataHandler, Planner
import wandb
import tqdm

def get_config_specific(value):
    config = deepcopy(BASE_CONFIG)
    if PROBLEM == 'sbm':
        p_inter = C-value/2
        p_outer = C+value/2
        config['data']['test']['problems'][PROBLEM]['p_inter'] = p_inter
        config['data']['test']['problems'][PROBLEM]['p_outer'] = p_outer
    elif PROBLEM in ('mcp', 'hhc'):
        config['data']['test']['problems'][PROBLEM][VALUE_NAME] = value
    else:
        raise NotImplementedError(f'Problem {PROBLEM} config modification not implemented.')
    return config

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Grid testing on the experiments from one W&B repository.')
    parser.add_argument('problem', metavar='problem', choices = ('mcp','hhc','sbm'), help='Need to choose an experiment')
    args = parser.parse_args()

    PROBLEM = args.problem
    print(f"Working on problem '{PROBLEM}'")
    if PROBLEM == 'mcp':
        VALUE_NAME = 'clique_size'
        VALUES = range(5,21)
    elif PROBLEM == 'sbm':
        VALUE_NAME = 'dc'
        VALUES = np.linspace(0,6,25)
        C=3
    elif PROBLEM == 'hhc':
        VALUE_NAME = 'fill_param'
        l_musquare = np.linspace(0,25,26)
        VALUES = np.sqrt(l_musquare)
    else:
        raise NotImplementedError(f"Problem {PROBLEM} not implemented.")

    ERASE_DATASETS = False
    ERASE_ARTIFACTS = True

    #WANDB
    WANDB_MODELS_PROJECT = f"repr_{PROBLEM}"
    WANDB_REPO_PROJECT = f"grid_{PROBLEM}"
    MODEL_VERSION = ':best'

    #VALUES_DEPENDING ON ABOVE
    BASE_PATH = 'scripts/'
    DATA_FILE = os.path.join(BASE_PATH, f'planner_files/recap_{PROBLEM}.csv')
    ADVANCE_LOG_FILE = os.path.join(BASE_PATH, f'planner_files/sweep_log_{PROBLEM}.csv')
    CONFIG_FILE_NAME = f'default_config.yaml'
    CONFIG_FILE = os.path.join(BASE_PATH, CONFIG_FILE_NAME)
    BASE_CONFIG = get_config(CONFIG_FILE)

    DH = DataHandler(DATA_FILE)
    planner = Planner(ADVANCE_LOG_FILE)

    wapi = wandb.Api()

    runs = wapi.runs(WANDB_MODELS_PROJECT)
    run_ids = [run.id for run in runs]
    total_runs = len(runs)

    test_loaders = []

    for value in VALUES:
        config = get_config_specific(value)
        test_loaders.append(get_test_dataset(config))
    
    for run in tqdm.tqdm(runs, total=len(runs)):
        pl_model = load_model(run.config, run.id, istest=True)
        trainer = setup_trainer(BASE_CONFIG, pl_model)
        trainer.test(pl_model, dataloaders=test_loaders)