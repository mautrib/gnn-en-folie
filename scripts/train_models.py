from copy import deepcopy
import sys, os
sys.path.append(os.getcwd())
from data import get_test_generator, get_train_val_generators
from toolbox.planner import Task, Planner
from commander import get_config, train, test
from pytorch_lightning import seed_everything
import wandb
import argparse
import numpy as np

def get_config_specific(value, config):
    config = deepcopy(config)
    if PROBLEM == 'sbm':
        p_inter = C-value/2
        p_outer = C+value/2
        config['data']['train']['problems'][PROBLEM]['p_inter'] = p_inter
        config['data']['train']['problems'][PROBLEM]['p_outer'] = p_outer
    elif PROBLEM in ('mcp', 'hhc'):
        config['data']['train']['problems'][PROBLEM][VALUE_NAME] = value
    else:
        raise NotImplementedError(f'Problem {PROBLEM} config modification not implemented.')
    return config

def erase_datasets(config):
    train_ds, val_ds = get_train_val_generators(config)
    train_ds.remove_files()
    val_ds.remove_files()

if __name__=='__main__':
    
    seed_everything(5344)
    
    parser = argparse.ArgumentParser(description='Grid testing on the experiments from one W&B repository.')
    parser.add_argument('problem', metavar='problem', choices = ('mcp','hhc','sbm'), help='Name of the experiment')
    parser.add_argument('--erase', metavar='erase', type=int, default=1, help='Set to 0 to conserve datasets.')
    parser.add_argument('--reset', metavar='reset', type=int, default=0, help='Set to 1 to erase all memory of values computed up to now.')
    args = parser.parse_args()
    
    RESET=bool(args.reset)
    ERASE_DATASETS=bool(args.erase)
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

    BASE_PATH = 'scripts/'
    CONFIG_FILE = 'train_models.yaml'
    PLANNER_FILE = 'planner_files/{PROBLEM}.csv'
    CONFIG_PATH = os.path.join(BASE_PATH, CONFIG_FILE)
    PLANNER_PATH = os.path.join(BASE_PATH, PLANNER_FILE)
    BASE_CONFIG = get_config(CONFIG_PATH)
    BASE_CONFIG['project'] = BASE_CONFIG['project'] + f'_{PROBLEM}'

    tasks = [Task(VALUE_NAME, value) for value in VALUES]

    planner = Planner(PLANNER_PATH)
    if RESET: planner.reset()
    planner.add_tasks(tasks)

    def step(config, task):
        config = get_config_specific(task.value, config)
        train(config)
        if ERASE_DATASETS: erase_datasets(config)
        wandb.finish()

    while planner.n_tasks!=0:
        task = planner.next_task()
        print(f"Task: {task}")
        step(BASE_CONFIG, task)
        planner.add_entry({task.column_name:task.value, "done":True})