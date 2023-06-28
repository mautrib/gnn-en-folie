from models.dgl_node import DGLNodeLoss
import dgl
import torch
import networkx as nx
from networkx.algorithms.approximation.clique import max_clique as nx_max_clique

from .base import UntrainableClass


class Networkx_Max_Clique(UntrainableClass):
    def __init__(self, batch_size=None, sync_dist=True):
        super().__init__(batch_size, sync_dist)
        self.loss = DGLNodeLoss(normalize=torch.nn.Identity())

    def forward(self, x: dgl.DGLGraph):
        assert torch.all(x.in_degrees() == x.out_degrees()), "Graph is not symmetric !"
        nxx = dgl.to_networkx(x)
        max_clique = nx_max_clique(nxx)
        proba = torch.zeros(x.number_of_nodes())
        proba[list(max_clique)] = 1
        final = torch.cat((1 - proba, proba), dim=-1)
        return final

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        g, target = batch
        raw_scores = self(g)
        loss_value = self.loss(raw_scores, target)
        self.log("test_loss", loss_value, sync_dist=self.sync_dist)
        self.log_metric("test", data=g, raw_scores=raw_scores, target=target)
        return loss_value
