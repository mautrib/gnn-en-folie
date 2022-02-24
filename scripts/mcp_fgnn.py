from copy import deepcopy
import sys, os
sys.path.append(os.getcwd())
from data import get_test_dataset, get_train_val_datasets
from toolbox.planner import Task, Planner
from commander import get_config, train, test
from pytorch_lightning import seed_everything
import wandb

seed_everything(5344)

ERASE_DATASETS=True
base_path = 'scripts'
planner_file = 'planner_files/mcp.csv'
config_file = 'mcp_fgnn.yaml'
config_path = os.path.join(base_path, config_file)
planner_path = os.path.join(base_path, planner_file)


l_clique_sizes = range(5,21)
tasks = [Task('clique_size', value) for value in l_clique_sizes]

planner = Planner(planner_path)
planner.add_tasks(tasks)

def erase_datasets(config):
    train_ds, val_ds = get_train_val_datasets(config)
    test_ds = get_test_dataset(config)
    train_ds.remove_files()
    val_ds.remove_files()
    test_ds.remove_files()

def step(planner):
    task = planner.next_task()
    print(f"Task: {task}")
    config = get_config(config_path)
    config['data']['train']['problems']['mcp']['clique_size'] = task.value
    config['data']['test']['problems']['mcp']['clique_size'] = task.value
    trainer = train(config)
    test(trainer, config)
    if ERASE_DATASETS: erase_datasets(config)
    wandb.finish()
    planner.add_entry({task.column_name:task.value, "done":True})


while planner.n_tasks!=0:
    step(planner)