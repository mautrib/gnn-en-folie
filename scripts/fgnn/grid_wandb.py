from copy import deepcopy
import sys, os
sys.path.append(os.getcwd())
import tqdm
from data import get_test_dataset, get_test_generator
from toolbox.planner import DataHandler, Planner
from commander import get_config, get_trainer_config, load_model, setup_trainer
import wandb
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.loggers import WandbLogger
wb_api = wandb.Api()
import argparse

def get_config_specific(value):
    config = deepcopy(BASE_CONFIG)
    if PROBLEM == 'sbm':
        p_inter = C-value/2
        p_outer = C+value/2
        config['data']['test']['problems'][PROBLEM]['p_inter'] = p_inter
        config['data']['test']['problems'][PROBLEM]['p_outer'] = p_outer
    elif PROBLEM in ('mcp', 'hhc'):
        config['data']['train']['problems'][PROBLEM][VALUE_NAME] = value
    else:
        raise NotImplementedError(f'Problem {PROBLEM} config modification not implemented.')
    return config

def setup_trainer(config):
    path = config['observers']['base_dir']
    path = os.path.join(os.getcwd(), path)
    trainer_config = get_trainer_config(config)
    wand_args = {
        'reinit': True
    }
    wandblogger = WandbLogger(project=WANDB_REPO_PROJECT, log_model="all", save_dir=path, **wand_args)
    trainer_config['logger'] = wandblogger
    trainer = pl.Trainer(**trainer_config)
    return trainer

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

    MODEL = 'fgnn'
    ERASE_DATASETS = False
    ERASE_ARTIFACTS = True

    #WANDB
    WANDB_ENTITY = 'mautrib'
    WANDB_MODELS_PROJECT = f"repr_{PROBLEM}"
    WANDB_REPO_PROJECT = f"grid_{PROBLEM}"
    MODEL_VERSION = ':best'

    #VALUES_DEPENDING ON ABOVE
    BASE_PATH = f'scripts/{MODEL}/'
    DATA_FILE = os.path.join(BASE_PATH, f'planner_files/recap_{PROBLEM}.csv')
    ADVANCE_LOG_FILE = os.path.join(BASE_PATH, f'planner_files/sweep_log_{PROBLEM}.csv')
    CONFIG_FILE_NAME = f'{PROBLEM}_{MODEL}.yaml'
    CONFIG_FILE = os.path.join(BASE_PATH, CONFIG_FILE_NAME)
    BASE_CONFIG = get_config(CONFIG_FILE)

    DH = DataHandler(DATA_FILE)
    planner = Planner(ADVANCE_LOG_FILE)

    runs = wb_api.runs(os.path.join(WANDB_ENTITY, WANDB_MODELS_PROJECT))
    total_runs = len(runs)

    test_loaders = []

    for value in VALUES:
        config = get_config_specific(value)
        test_loaders.append(get_test_dataset(config))

    art = wandb.init(project=WANDB_REPO_PROJECT, reinit=True)
    for run in tqdm.tqdm(runs, total=total_runs):
        model_name = f'model-{run.id}{MODEL_VERSION}'
        model_artifact_name = os.path.join(WANDB_ENTITY, WANDB_MODELS_PROJECT, model_name)
        print(f"Getting model artifact from : {model_artifact_name}")
        artifact = art.use_artifact(model_artifact_name, 'model')
        art_dir = artifact.download()
        model_dir = os.path.join(art_dir, 'model.ckpt')
        pl_model = load_model(BASE_CONFIG, model_dir)
        trainer = setup_trainer(config)
        trainer.test(pl_model, dataloaders=test_loaders)
        summary = get_values(trainer)
        run = trainer.logger.experiment
        run.summary['values_logged'] = summary
        wandb.finish()
        if ERASE_ARTIFACTS:
            os.remove(model_dir)
            os.rmdir(art_dir)
    if ERASE_ARTIFACTS:
        os.rmdir(os.path.join(BASE_PATH, 'artifacts'))

    if ERASE_DATASETS:
        for value in VALUES:
            config = deepcopy(BASE_CONFIG)
            config['data']['train']['problems'][PROBLEM][VALUE_NAME] = value
            test_generator = get_test_generator(config)
            test_generator.remove_files()

