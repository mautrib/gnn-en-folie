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

PROBLEM='sbm'
VALUE_NAME = 'dc'
C = 3

ERASE_DATASETS=True
base_path = 'scripts/fgnn'
planner_file = f'planner_files/{PROBLEM}.csv'
config_file = f'{PROBLEM}_fgnn.yaml'
config_path = os.path.join(base_path, config_file)
planner_path = os.path.join(base_path, planner_file)


l_dc = np.linspace(0,6,25)

tasks = [Task(VALUE_NAME, value) for value in l_dc]

planner = Planner(planner_path)
planner.add_tasks(tasks)

def erase_datasets(config):
    train_ds, val_ds = get_train_val_generators(config)
    test_ds = get_test_generator(config)
    train_ds.remove_files()
    val_ds.remove_files()
    test_ds.remove_files()

def step(config, task):
    p_inter = C-task.value/2
    p_outer = C+task.value/2
    config['data']['train']['problems'][PROBLEM]['p_inter'] = p_inter
    config['data']['test']['problems'][PROBLEM]['p_inter'] = p_inter
    config['data']['train']['problems'][PROBLEM]['p_outer'] = p_outer
    config['data']['test']['problems'][PROBLEM]['p_outer'] = p_outer
    trainer = train(config)
    test(trainer, config)
    if ERASE_DATASETS: erase_datasets(config)
    wandb.finish()

while planner.n_tasks!=0:
    task = planner.next_task()
    print(f"Task: {task}")
    config = get_config(config_path)
    step(config, task)
    planner.add_entry({task.column_name:task.value, "done":True})