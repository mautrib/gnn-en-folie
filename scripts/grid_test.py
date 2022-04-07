import os, sys
sys.path.append(os.getcwd())
from copy import deepcopy
import argparse
import numpy as np
import wandb
import tqdm

from data import get_test_dataset
from commander import get_config, load_model, setup_trainer
from metrics import setup_metric
from toolbox.planner import DataHandler, Planner

def get_config_specific(value, config=None):
    if config is None: config = BASE_CONFIG
    config = deepcopy(config)
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

def get_values(trainer):
    logged = trainer.logged_metrics
    loss_name = 'test_loss/dataloader_idx_{}'
    metrics_name = 'test.metrics/dataloader_idx_{}'
    total_dict = {}
    for i, value in enumerate(VALUES):
        loss_value = logged[loss_name.format(i)]
        metrics_value = logged[metrics_name.format(i)]
        values_dict = {
            'loss': loss_value,
            'metrics': metrics_value
        }
        total_dict[f'{value:.4f}'] = values_dict
    return total_dict

def get_train_value(run):
    config = run.config
    if PROBLEM == 'sbm':
        p_outer = config['data']['train']['problems'][PROBLEM]['p_outer']
        p_inter = config['data']['train']['problems'][PROBLEM]['p_inter']
        value = p_outer-p_inter
    elif PROBLEM in ('mcp', 'hhc'):
        value = config['data']['train']['problems'][PROBLEM][VALUE_NAME]
    else:
        raise NotImplementedError(f'Problem {PROBLEM} config modification not implemented.')
    return value

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Grid testing on the experiments from one W&B repository.')
    parser.add_argument('problem', metavar='problem', choices = ('mcp','hhc','sbm'), help='Need to choose an experiment')
    parser.add_argument('--skip', metavar='skip', type=int, default=0, help='Number of experiments to skip')
    args = parser.parse_args()

    SKIP_FIRST_N_RUNS=args.skip
    PROBLEM = args.problem
    
    print(f"Working on problem '{PROBLEM}'")
    if PROBLEM == 'mcp':
        VALUE_NAME = 'clique_size'
        VALUES = range(5,20)
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

    #WANDB
    WANDB_MODELS_PROJECT = f"trained_models_{PROBLEM}"

    #VALUES_DEPENDING ON ABOVE
    BASE_PATH = 'scripts/'
    DATA_FILE = os.path.join(BASE_PATH, f'planner_files/recap_{PROBLEM}.csv')
    ADVANCE_LOG_FILE = os.path.join(BASE_PATH, f'planner_files/sweep_log_{PROBLEM}.csv')
    CONFIG_FILE_NAME = f'grid_config.yaml'
    CONFIG_FILE = os.path.join(BASE_PATH, CONFIG_FILE_NAME)
    BASE_CONFIG = get_config(CONFIG_FILE)

    DH = DataHandler(DATA_FILE)
    planner = Planner(ADVANCE_LOG_FILE)

    wapi = wandb.Api()

    runs = wapi.runs(WANDB_MODELS_PROJECT)
    run_ids = [run.id for run in runs]
    total_runs = len(runs)

    test_loaders = []

    if SKIP_FIRST_N_RUNS!=0: print(f'Skipping first {SKIP_FIRST_N_RUNS} runs.')
    run_number=-1

    for run in tqdm.tqdm(runs, total=len(runs)):
        run_number+=1
        if run_number < SKIP_FIRST_N_RUNS: continue
        pl_model = load_model(run.config, run.id, add_metric=False)
        test_loaders = []
        for value in VALUES:
            config = get_config_specific(value)
            config['arch'] = run.config['arch'] #So that fgnn keep fgnn data and dgl keep using dgl
            test_loaders.append(get_test_dataset(config))
        setup_metric(pl_model, BASE_CONFIG, istest=True)
        trainer = setup_trainer(BASE_CONFIG, pl_model, only_test=True)
        trainer.test(pl_model, dataloaders=test_loaders)
        run_id = trainer.logger.experiment.id #First store the id
        wandb.finish()
        #Update the summary after stopping the ddp experiment to prevent some trouble
        project_path = os.path.join(BASE_CONFIG['project'] + f'_{PROBLEM}', run_id)
        run_copy = wapi.run(project_path)
        run_copy.summary['train_value'] = get_train_value(run)
        run_copy.summary['values'] = [f"{value:.4f}" for value in VALUES]
        run_copy.summary['logged'] = trainer.logged_metrics
        run_copy.summary.update()