from models.fgnn_edge import FGNN_Edge
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

pl.seed_everything(2009)

def main():
    args_dict = {'original_features_num':2, 
                    'num_blocks':2, 
                    'in_features': 32,
                    'out_features': 1,
                    'depth_of_mlp': 3}
    model = FGNN_Edge(args_dict)
    wandb_logger = WandbLogger(project="gnnpl", log_model="all")
    trainer = pl.Trainer(logger=wandb_logger, max_epochs=100)
    wandb_logger.watch(model)
    from data.tsp import TSP_Generator
    data_args = {"num_examples_train":10,"num_examples_val":10,"path_dataset":"dataset_test","n_vertices":50, 'distance_used':'EUC_2D','generative_model':'Square01'}
    train_loader = TSP_Generator('train', data_args)
    train_loader.load_dataset(use_dgl=False)
    val_loader = TSP_Generator('val', data_args)
    val_loader.load_dataset(use_dgl=False)

    tloader = DataLoader(train_loader, batch_size = 1, num_workers=8)
    vloader = DataLoader(val_loader, batch_size = 1, num_workers=8)

    trainer.fit(model, tloader, vloader)

if __name__=="__main__":
    main()