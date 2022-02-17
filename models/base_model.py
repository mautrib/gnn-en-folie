import pytorch_lightning as pl
import torch.optim

class GNN_Abstract_Base_Class(pl.LightningModule):

    def __init__(self, model, optim_args):
        super().__init__()
        self.model = model
        self.initial_lr = optim_args['lr']
        self.scheduler_args = {
            'patience': optim_args['scheduler_step'],
            'factor': optim_args['scheduler_factor']
        }
        self.scheduler_monitor = optim_args['monitor']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, **(self.scheduler_args))
        return {'optimizer':optimizer, 'lr_scheduler': scheduler, 'monitor': self.scheduler_monitor}
