from copy import deepcopy
import sys, os
sys.path.append(os.getcwd())
from data import get_test_generator, get_train_val_generators
from toolbox.planner import Task, Planner
from commander import get_config, train, test
from pytorch_lightning import seed_everything
import wandb
import numpy as np

seed_everything(5344)

PROBLEM='hhc'
VALUE_NAME = 'fill_param'

ERASE_DATASETS=True
base_path = 'scripts'
planner_file = f'planner_files/{PROBLEM}.csv'
config_file = f'{PROBLEM}_fgnn.yaml'
config_path = os.path.join(base_path, config_file)
planner_path = os.path.join(base_path, planner_file)


l_musquare = np.linspace(0,25,26)
l_mus = np.sqrt(l_musquare)
tasks = [Task(VALUE_NAME, value) for value in l_mus]

planner = Planner(planner_path)
planner.add_tasks(tasks)

def erase_datasets(config):
    train_ds, val_ds = get_train_val_generators(config)
    test_ds = get_test_generator(config)
    train_ds.remove_files()
    val_ds.remove_files()
    test_ds.remove_files()

def step(planner):
    task = planner.next_task()
    print(f"Task: {task}")
    config = get_config(config_path)
    config['data']['train']['problems'][PROBLEM][VALUE_NAME] = task.value
    config['data']['test']['problems'][PROBLEM][VALUE_NAME] = task.value
    trainer = train(config)
    test(trainer, config)
    if ERASE_DATASETS: erase_datasets(config)
    wandb.finish()
    planner.add_entry({task.column_name:task.value, "done":True})


while planner.n_tasks!=0:
    step(planner)