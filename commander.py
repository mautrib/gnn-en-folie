import yaml
import toolbox.utils as utils
import os

from models import get_pipeline, get_pl_model, get_torch_model, get_optim_args
from data import get_test_dataset, get_train_val_datasets
from metrics import setup_metric
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import argparse

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
        logger = WandbLogger(project=f"{config['project']}_{config['problem']}", log_model="all", save_dir=path)
        logger.experiment.config.update(config)
    else:
        raise NotImplementedError(f"Observer {observer} not implemented.")
    return logger

def load_model(config, path):
    path = config['train']['start_model']
    print(f'Loading base model from {path}... ', end = "")
    PL_Model_Class = get_pl_model(config)
    pl_model = PL_Model_Class.load_from_checkpoint(path, model=get_torch_model(config), optim_args=get_optim_args(config))
    print('Done.')
    return pl_model

def get_trainer_config(config):
    trainer_config = config['train']
    accelerator_config = utils.get_accelerator_dict(config['device'])
    trainer_config.update(accelerator_config)
    early_stopping = EarlyStopping('lr', verbose=True, mode='max', patience=1+config['train']['max_epochs'], divergence_threshold=config['train']['optim_args']['lr_stop'])
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, verbose=True)
    trainer_config['callbacks'] = [early_stopping, checkpoint_callback]
    clean_config = utils.restrict_dict_to_function(pl.Trainer.__init__, trainer_config)
    return clean_config

def setup_trainer(config, model):
    trainer_config = get_trainer_config(config)
    if config['observers']['use']:
        logger = get_observer(config)
        logger.watch(model)
        trainer_config['logger'] = logger
    trainer = pl.Trainer(**trainer_config)
    return trainer

def train(config):
    if config['train']['anew']:
        pl_model = get_pipeline(config)
    else:
        pl_model = load_model(config, config['train']['start_model'])
    setup_metric(pl_model, config)
    trainer = setup_trainer(config, pl_model)
    train_dataset, val_dataset = get_train_val_datasets(config)
    trainer.fit(pl_model, train_dataset, val_dataset)
    return trainer

def test(trainer, config):
    test_dataset = get_test_dataset(config)
    arg_dict = {'dataloaders': test_dataset,
                'verbose':True
    }
    if trainer is None:
        pl_model = load_model(config, config['train']['start_model'])
        trainer = pl.Trainer()
        arg_dict['model'] = pl_model
    else:
        arg_dict['ckpt_path'] = 'best'
    trainer.test(**arg_dict)


def main():
    parser = argparse.ArgumentParser(description='Main file for creating experiments.')
    parser.add_argument('command', metavar='c', choices=['train','test'],
                    help='Command to execute : train or test')
    args = parser.parse_args()
    if args.command=='train':
        training=True
        default_test = False
    elif args.command=='test':
        training=False
        default_test=True
    
    config = get_config()
    config = utils.clean_config(config)
    trainer=None
    if training:
        trainer = train(config)
    if default_test or config['test_enabled']:
        test(trainer, config)

if __name__=="__main__":
    pl.seed_everything(3787, workers=True)
    main()