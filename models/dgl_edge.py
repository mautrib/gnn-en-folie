import pytorch_lightning as pl
import torch
from models.base_model import GNN_Abstract_Base_Class

class DGLEdgeLoss(torch.nn.Module):
    def __init__(self, normalize=torch.nn.Sigmoid(), loss=torch.nn.CrossEntropyLoss(reduction='mean')):
        super(DGLEdgeLoss, self).__init__()
        self.loss = loss
        self.normalize = normalize

    def forward(self, raw_scores, target):
        target = target.edata['solution']
        preds = self.normalize(raw_scores)
        loss = self.loss(preds,target)
        return torch.mean(loss)

class DGL_Edge(GNN_Abstract_Base_Class):
    
    def __init__(self, model, optim_args):
        super().__init__(model, optim_args)
        self.loss = DGLEdgeLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        g, target = batch
        x = self(g)
        loss_value = self.loss(x, target)
        self.log('train_loss', loss_value)
        return loss_value
    
    def validation_step(self, batch, batch_idx):
        g, target = batch
        x = self(g)
        loss_value = self.loss(x, target)
        self.log('val_loss', loss_value)
        return loss_value