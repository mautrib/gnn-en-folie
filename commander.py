import yaml
import toolbox.utils as utils
import os

from models import get_pipeline
from data import get_train_val_datasets
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

def get_config(filename='default_config.yaml'):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_observer(config):
    path = config['observers']['base_dir']
    path = os.path.join(os.getcwd(), path)
    utils.check_dir(path)
    observer = config['observers']['observer']
    if observer=='wandb':
        logger = WandbLogger(project=config['name'], log_model="all", save_dir=path)
        logger.experiment.config.update(config)
    else:
        raise NotImplementedError(f"Observer {observer} not implemented.")
    return logger

def get_trainer_config(config):
    trainer_config = config['train']
    accelerator_config = utils.get_accelerator_dict(config['device'])
    trainer_config.update(accelerator_config)
    clean_config = utils.restrict_dict_to_function(pl.Trainer.__init__, trainer_config)
    return clean_config

def setup_trainer(config, model):
    logger = get_observer(config)
    logger.watch(model)
    trainer_config = get_trainer_config(config)
    trainer = pl.Trainer(logger=logger, **trainer_config)
    return trainer

def train(config):
    pl_model = get_pipeline(config)
    trainer = setup_trainer(config, pl_model)
    train_dataset, val_dataset = get_train_val_datasets(config)
    trainer.fit(pl_model, train_dataset, val_dataset)

def main():
    config = get_config()
    config = utils.clean_config(config)
    train(config)

if __name__=="__main__":
    main()