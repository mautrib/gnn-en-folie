import pytorch_lightning as pl
import torch
from models.fgnn import Simple_Edge_Embedding

class FGNN_Edge(pl.LightningModule):
    
    def __init__(self, args_dict, normalize=torch.nn.Sigmoid(), loss=torch.nn.CrossEntropyLoss(reduction='mean')):
        super().__init__()
        self.model = Simple_Edge_Embedding(**args_dict)
        self.normalize = normalize
        self.loss = loss
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        g, target = batch
        x = self(g)
        probas = self.normalize(x)
        loss_value = self.loss(probas, target)
        self.log('train_loss', loss_value)
        return loss_value
    
    def validation_step(self, batch, batch_idx):
        g, target = batch
        x = self(g)
        probas = self.normalize(x)
        loss_value = self.loss(probas, target)
        self.log('val_loss', loss_value)
        return loss_value

    def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer