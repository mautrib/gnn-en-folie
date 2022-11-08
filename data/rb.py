
import os, sys
print(os.getcwd())
sys.path.append(os.getcwd())
from data.base import Base_Generator

import dgl
import tqdm
from data.mcp import MCP_Generator
from toolbox.conversions import adjacency_matrix_to_tensor_representation, connectivity_to_dgl, dense_tensor_to_edge_format

from toolbox.searches.mcp.mcp_solver import MCP_Solver

RB_GRAPH_PATH = os.path.join("data","rb_data", "dgl_graph.bin")

class RB_Generator(Base_Generator):
    def __init__(self, name, args):
        self.num_examples=500
        path_dataset = os.path.join(args['path_dataset'],"RB")
        self.n_threads = args.get('n_threads', 24)
        super().__init__(name, path_dataset, self.num_examples)

    def create_dataset(self):
        graphs, clique_numbers = dgl.load_graphs(RB_GRAPH_PATH)
        self.n_vertices = [graph.num_nodes() for graph in graphs]
        self.clique_numbers = clique_numbers
        l_adjacency = [g.adj().to_dense() for g in graphs]
        solver = MCP_Solver(l_adjacency, max_threads=self.n_threads)
        solver.solve(force_list=True)
        assert len(solver.solutions)==self.num_examples, "Error somewhere in MCP solver."
        clique_solutions = [solution[0] for solution in solver.solutions]
        l_b = [adjacency_matrix_to_tensor_representation(W) for W in l_adjacency]
        l_k = [MCP_Generator.mcp_ind_to_adj(elt, n_vertices) for elt,n_vertices in zip(clique_solutions,self.n_vertices)]
        l_k = [self._solution_conversion(K.to(int), connectivity_to_dgl(B)) for (B,K) in zip(l_b,l_k)]
        l_data = [(B,K) for (B,K) in zip(l_b,l_k)]
        return l_data
    
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

if __name__=="__main__":
    rbg = RB_Generator('test',{'path_dataset':'datasets', "n_threads":2})
    rbg.load_dataset()
