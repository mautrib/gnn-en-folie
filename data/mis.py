from data.base import Base_Generator
from data.graph_generation import GENERATOR_FUNCTIONS
from toolbox.conversions import (
    adjacency_matrix_to_tensor_representation,
    connectivity_to_dgl,
    dense_tensor_to_edge_format,
)
from toolbox.searches.mcp import mcp_beam_method
import os
import torch
import toolbox.utils as utils
import random
import dgl
from networkx.algorithms.approximation.clique import max_clique
import tqdm


# Now we create the same class as HHC_Generator, but instead of the HHC, use the independent set problem, name it MIS_Generator
class MIS_Generator(Base_Generator):
    """
    Independent Set Generator
    """

    def __init__(self, name, args):
        self.edge_density = args["edge_density"]
        self.is_size = int(args["is_size"])
        num_examples = args["num_examples_" + name]
        self.n_vertices = args["n_vertices"]
        subfolder_name = "MIS_{}_{}_{}_{}".format(
            num_examples, self.n_vertices, self.is_size, self.edge_density
        )
        path_dataset = os.path.join(
            args["path_dataset"], "mis", subfolder_name
        )  # Change the path_dataset to 'mis'
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        self.constant_n_vertices = True
        utils.check_dir(self.path_dataset)

    def _plant_is(self, W):
        """Plants an independent set"""
        assert isinstance(
            self.is_size, int
        ), f"Independent set size is not an int {self.is_size=}"
        W, K = self.add_is_base(W, self.is_size)
        B = adjacency_matrix_to_tensor_representation(W)
        return (B, K)

    @staticmethod
    def add_is_base(W, k):
        K = torch.zeros((len(W), len(W)))
        K[:k, :k] = torch.ones((k, k))  # Solution is these K nodes
        W[:k, :k] = torch.zeros((k, k))  # Remove edges between these nodes
        return W, K
    
    def compute_example(self):
        """Compute a single example"""
        try:
            g, W = GENERATOR_FUNCTIONS["ErdosRenyi"](self.edge_density, self.n_vertices)
        except KeyError:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        B,K = self._plant_is(W)
        K = K.to(int)
        K = self._solution_conversion(K, connectivity_to_dgl(B))
        return (B,K)
    
    @classmethod
    def _solution_conversion(cls, target, dgl_graph):
        num_nodes = dgl_graph.num_nodes()
        target_dgl = dgl.graph(dgl_graph.edges(), num_nodes=num_nodes)
        target_dgl = dgl.add_self_loop(target_dgl)
        edge_classif = dense_tensor_to_edge_format(target, target_dgl)
        node_classif = (target.sum(dim=-1)!=0).to(target.dtype) # Get a node classification of shape (N)
        node_classif = node_classif.unsqueeze(-1) # Modify it to size (N,1)
        target_dgl.edata['solution'] = edge_classif
        target_dgl.ndata['solution'] = node_classif
        return target_dgl
