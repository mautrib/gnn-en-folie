from data.base import Base_Generator
from toolbox import utils
import torch
import os
import dgl
from toolbox.conversions import adjacency_matrix_to_tensor_representation, dense_tensor_to_edge_format

class SBM_Generator(Base_Generator):
    def __init__(self, name, args):
        self.n_vertices = args['n_vertices']
        self.p_inter = args['p_inter']
        self.p_outer = args['p_outer']
        self.pi_real = self.p_inter/self.n_vertices
        self.po_real = self.p_outer/self.n_vertices
        self.alpha = args['alpha']
        num_examples = args['num_examples_' + name]
        subfolder_name = 'SBM_{}_{}_{}_{}_{}'.format(num_examples,
                                                           self.n_vertices,
                                                           self.alpha, 
                                                           self.p_inter,
                                                           self.p_outer)
        path_dataset = os.path.join(args['path_dataset'], 'sbm',
                                         subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        self.constant_n_vertices = True
        utils.check_dir(self.path_dataset)
    
    def compute_example(self):
        """
        Computes the pair (data,target). Data is the adjacency matrix. For target, there are 2 interpretations :
         - if used with similarity : K_{ij} is 1 if node i and j are from the same group
         - if used with edge probas: K_{ij} is 1 if it's an intra-edge (so i and j from the same group)
        """
        n = self.n_vertices
        n_sub_a = n//2
        n_sub_b = n - n_sub_a # In case n_vertices is odd


        ga = torch.empty((n_sub_a,n_sub_a)).uniform_()
        ga = (ga<(self.pi_real)).to(torch.float)
        gb = torch.empty((n_sub_b,n_sub_b)).uniform_()
        gb = (gb<(self.pi_real)).to(torch.float)
        ga = utils.symmetrize_matrix(ga)
        gb = utils.symmetrize_matrix(gb)

        glink = (torch.empty((n_sub_a,n_sub_b)).uniform_()<self.po_real).to(torch.float)
        
        adj = torch.zeros((self.n_vertices,self.n_vertices))

        adj[:n_sub_a,:n_sub_a] = ga.detach().clone()
        adj[:n_sub_a,n_sub_a:] = glink.detach().clone()
        adj[n_sub_a:,:n_sub_a] = glink.T.detach().clone()
        adj[n_sub_a:,n_sub_a:] = gb.detach().clone()


        K = torch.zeros((n,n), dtype=int)
        K[:n_sub_a,:n_sub_a] = 1
        K[n_sub_a:,n_sub_a:] = 1
        #K,adj = utils.permute_adjacency_twin(K,adj)
        B = adjacency_matrix_to_tensor_representation(adj)
        return (B, K)
    
    @classmethod
    def _solution_conversion(cls, target, dgl_graph):
        num_nodes = dgl_graph.num_nodes()
        target_dgl = dgl.graph(dgl_graph.edges(), num_nodes=num_nodes)
        edge_classif = dense_tensor_to_edge_format(target, target_dgl)
        node_classif = target[0,:] #Keep the node 0 to a class 1
        node_classif = node_classif.unsqueeze(-1) # Modify it to size (N,1)
        target_dgl.edata['solution'] = edge_classif
        target_dgl.ndata['solution'] = node_classif
        return target_dgl