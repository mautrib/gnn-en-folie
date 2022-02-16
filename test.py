from models.gatedgcn import GatedGCNNet_Edge
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from dgl import batch as dglbatch
from pytorch_lightning.loggers import WandbLogger

pl.seed_everything(2009)

def _collate_fn_dgl_ne(samples_list):
    bs = len(samples_list)
    input1_list = [input1 for (input1, _) in samples_list]
    target_list = [target for (_,target) in samples_list]
    shape = target_list[0].shape
    input_batch = dglbatch(input1_list)
    target_batch = torch.zeros((bs,)+ shape, dtype=target_list[0].dtype)
    #print("Shape : ",target_list[0].shape)
    for i,target in enumerate(target_list):
        target_batch[i] = target
    if bs==1:
        target_batch = target_batch.squeeze(0)
    return (input_batch,target_batch)

def main():
    model = GatedGCNNet_Edge()
    wandb_logger = WandbLogger(project="gnnpl", log_model="all")
    trainer = pl.Trainer(devices=1,accelerator="gpu",logger=wandb_logger, max_epochs=100)
    wandb_logger.watch(model)
    from data.tsp import TSP_Generator
    data_args = {"num_examples_train":10000,"num_examples_val":1000,"path_dataset":"dataset_test","n_vertices":50, 'distance_used':'EUC_2D','generative_model':'Square01'}
    train_loader = TSP_Generator('train', data_args)
    train_loader.load_dataset(use_dgl=True)
    val_loader = TSP_Generator('val', data_args)
    val_loader.load_dataset(use_dgl=True)

    tloader = DataLoader(train_loader, batch_size = 1, collate_fn=_collate_fn_dgl_ne, num_workers=8)
    vloader = DataLoader(val_loader, batch_size = 1, collate_fn=_collate_fn_dgl_ne, num_workers=8)

    trainer.fit(model, tloader, vloader)

if __name__=="__main__":
    main()