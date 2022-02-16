import pytorch_lightning as pl
import torch
from models.fgnn import Simple_Edge_Embedding

class EdgeClassifLoss(torch.nn.Module):
    def __init__(self, normalize=torch.nn.Sigmoid(), loss=torch.nn.BCELoss(reduction='mean')):
        super(EdgeClassifLoss, self).__init__()
        if isinstance(loss, torch.nn.BCELoss):
            self.loss = lambda preds,target: loss(preds, target.to(torch.float))
        else:
            self.loss = loss
        self.normalize = normalize

    def forward(self, raw_scores, target):
        """
        outputs is the output of siamese network (bs,n_vertices,n_vertices)
        """
        preds = self.normalize(raw_scores)
        loss = self.loss(preds,target)
        return torch.mean(loss)

class FGNN_Edge(pl.LightningModule):
    
    def __init__(self, args_dict, normalize=torch.nn.Sigmoid(), loss=torch.nn.BCELoss(reduction='mean')):
        super().__init__()
        self.model = Simple_Edge_Embedding(**args_dict)
        self.loss = EdgeClassifLoss(normalize, loss)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        g, target = batch
        x = self(g)
        probas = probas.squeeze(-1)
        loss_value = self.loss(probas, target)
        self.log('train_loss', loss_value)
        return loss_value
    
    def validation_step(self, batch, batch_idx):
        g, target = batch
        x = self(g)
        probas = probas.squeeze(-1)
        loss_value = self.loss(probas, target)
        self.log('val_loss', loss_value)
        return loss_value

    def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer