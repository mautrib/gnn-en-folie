from copy import deepcopy
import sys, os
sys.path.append(os.getcwd())
import tqdm
from data import get_test_dataset
from toolbox.planner import DataHandler
from commander import get_config, load_model, setup_metric, setup_trainer
import wandb
wb_api = wandb.Api()
import numpy as np

#BASE VALUES
PROBLEM = 'hhc'
VALUE_NAME = 'fill_param'
MODEL = 'fgnn'

#WANDB
WANDB_ENTITY = 'mautrib'
WANDB_PROJECT = f"repr_{PROBLEM}"

MODELS_DIR = "/home/mautrib/phd/gnn-en-folie/observers/test_mcp/11q1vfjp/" #f'observers/repr_{PROBLEM}/'
BASE_PATH = f'scripts/{MODEL}/'
DATA_FILE = os.path.join(BASE_PATH, f'planner_files/recap_{PROBLEM}.csv')
CONFIG_FILE_NAME = f'{PROBLEM}_{MODEL}.yaml'
CONFIG_FILE = os.path.join(BASE_PATH, CONFIG_FILE_NAME)
BASE_CONFIG = get_config(CONFIG_FILE)

ERASE_DATASETS = True

l_musquare = np.linspace(0,25,26)
VALUES = np.sqrt(l_musquare)

DH = DataHandler(DATA_FILE)

def get_exp_name_from_ckpt_path(ckpt_path):
    name,_ = ckpt_path.split('/checkpoints/')
    name = name[-8:]
    return name

def get_train_value_hhc(base_exp_name):
    run = wb_api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{base_exp_name}")
    config = run.config
    value = config['data']['train']['problems'][PROBLEM][VALUE_NAME]
    return value

def handle_data(data_dict):
    handled = {}
    handled.update({'loss': data_dict['test_loss']})
    handled.update({'accuracy': data_dict['test.metrics']['accuracy']})
    # handled.update({'f1': data_dict['test.metrics']['f1']})
    # handled.update({'recall': data_dict['test.metrics']['recall']})
    # handled.update({'precision': data_dict['test.metrics']['precision']})
    return handled

def prepare_dataset(config, value):
    config['data']['test']['problems']['hhc']['fill_param'] = value
    test_dataset = get_test_dataset(config)
    return test_dataset

def step(path_to_chkpt):
    config = deepcopy(BASE_CONFIG)
    config['project'] = 'sweep'
    base_exp_name = get_exp_name_from_ckpt_path(path_to_chkpt)
    train_value = get_train_value_hhc(base_exp_name)
    pl_model = load_model(config, path_to_chkpt)
    setup_metric(pl_model, config)
    progress_bar = tqdm.tqdm(VALUES)
    for value in progress_bar:
        progress_bar.set_description(f'Value {value}.')
        test_dataset = prepare_dataset(config, value)
        trainer = setup_trainer(config, pl_model, watch=False)
        wandblogger = trainer.logger.experiment
        wandblogger.config['train_value'] = train_value
        trainer.test(pl_model, test_dataloaders=test_dataset)
        wandb.finish()
        logged_metrics = trainer.logged_metrics
        clean_metrics = handle_data(logged_metrics)
        clean_metrics[VALUE_NAME] = value
        clean_metrics['train_value'] = train_value
        DH.add_entry(clean_metrics)


to_walk = os.walk(MODELS_DIR)
to_walk = [elt for elt in to_walk]
for dirpath, _, filenames in tqdm.tqdm(to_walk):
    for filename in filenames:
        path_to_file = os.path.join(dirpath, filename)
        step(path_to_chkpt=path_to_file)
        

        
