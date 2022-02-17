import torch
import os
import tqdm
import dgl
from numpy import mgrid as npmgrid, argpartition as npargpartition
from toolbox.utils import is_adj

def sparsify_adjacency(adjacency, sparsify, distances):
    assert is_adj(adjacency), "Matrix is not an adjacency matrix"
    assert isinstance(sparsify,int), f"Sparsify not recognized. Should be int (number of closest neighbors), got {sparsify=}"
    assert distances.shape == adjacency.shape, f"Distances of different shape than adjacency {distances.shape}!={adjacency.shape}"
    N,_ = adjacency.shape
    mask = torch.zeros_like(adjacency)
    knns = npargpartition(distances, kth=sparsify, axis=-1)[:, sparsify ::-1].copy()
    range_tensor = torch.tensor(range(N)).unsqueeze(-1)
    mask[range_tensor,knns,1] = 1
    mask = mask*(1-torch.eye(N)) #Remove the self value
    return adjacency*mask

def _adjacency_to_dgl(adj):
    assert is_adj(adj), "Matrix is not an adjacency matrix"
    N,_ = adj.shape
    mgrid = npmgrid[:N,:N].transpose(1,2,0)
    edges = mgrid[torch.where(adj==1)]
    edges = edges.T #To have the shape (2,n_edges)
    src,dst = [elt for elt in edges[0]], [elt for elt in edges[1]] #DGLGraphs don't like Tensors as inputs...
    gdgl = dgl.graph((src,dst),num_nodes=N)
    return gdgl

def dense_tensor_to_edge_format(dense_tensor: torch.Tensor, dgl_graph: dgl.graph):
    assert dense_tensor.dim()==2 and dense_tensor.shape[0]==dense_tensor.shape[1], f"Dense Tensor isn't of shape (N,N)"
    N,_ = dense_tensor.shape
    src,rst = dgl_graph.edges()
    edge_tensor = dense_tensor[src,rst]
    return edge_tensor.unsqueeze(-1)

def connectivity_to_dgl(connectivity_graph, sparsify=None, distances = None):
    assert connectivity_graph.dim()==3, "Tensor dimension not recognized. Should be (N_nodes, N_nodes, N_features)"
    N, _, N_feats = connectivity_graph.shape
    degrees = connectivity_graph[:,:,0]
    assert torch.all(degrees*(1-torch.eye(N))==0) and not torch.any(degrees.diag() != degrees.diag().to(int)), "Tensor first feature isn't degree. Not recognized."
    if N_feats == 2:
        edge_features = connectivity_graph[:,:,1]
        adjacency = (edge_features!=0).to(torch.float)
        if sparsify is not None:
            if distances is None:
                distances = edge_features
            adjacency = sparsify_adjacency(adjacency, sparsify, distances)
        gdgl = _adjacency_to_dgl(adjacency)
        gdgl.ndata['feat'] = degrees.diagonal().reshape((N,1)) #Adding degrees to node features
        if not is_adj(edge_features):
            #src,rst = gdgl.edges() #For now only contains node features
            #efeats = edge_features[src,rst]
            efeats = dense_tensor_to_edge_format(edge_features, gdgl)
            gdgl.edata["feat"] = efeats
    else:
        raise NotImplementedError("Haven't implemented the function for more Features than one per edge (problem is how to check for the adjacency matrix).")
    return gdgl

def adjacency_matrix_to_tensor_representation(W):
    """ Create a tensor B[:,:,1] = W and B[i,i,0] = deg(i)"""
    degrees = W.sum(1)
    B = torch.zeros((len(W), len(W), 2))
    B[:, :, 1] = W
    indices = torch.arange(len(W))
    B[indices, indices, 0] = degrees
    return B


class Base_Generator(torch.utils.data.Dataset):
    def __init__(self, name, path_dataset, num_examples):
        self.path_dataset = path_dataset
        self.name = name
        self.num_examples = num_examples

    def load_dataset(self, use_dgl=False):
        """
        Look for required dataset in files and create it if
        it does not exist
        """
        filename = self.name + '.pkl'
        filename_dgl = self.name + '_dgl.pkl'
        path = os.path.join(self.path_dataset, filename)
        path_dgl = os.path.join(self.path_dataset, filename_dgl)
        if os.path.exists(path):
            if use_dgl:
                print('Reading dataset at {}'.format(path_dgl))
                data = torch.load(path_dgl)
            else:
                print('Reading dataset at {}'.format(path))
                data = torch.load(path)
            self.data = list(data)
        else:
            print('Creating dataset at {}'.format(path))
            l_data = self.create_dataset()
            print('Saving dataset at {}'.format(path))
            torch.save(l_data, path)
            print('Creating dataset at {}'.format(path_dgl))
            print("Converting data to DGL format")
            l_data_dgl = []
            for data,target in tqdm.tqdm(l_data):
                elt_dgl = connectivity_to_dgl(data)
                target_dgl = self._solution_conversion(target, elt_dgl)
                l_data_dgl.append((elt_dgl,target_dgl))
            print("Conversion ended.")
            print('Saving dataset at {}'.format(path_dgl))
            torch.save(l_data_dgl, path_dgl)
            if use_dgl:
                self.data = l_data_dgl
            else:
                self.data = l_data
    
    @staticmethod
    def _solution_conversion(target, dgl_graph):
        raise NotImplementedError("This function should be implemented in your generator. Base_Generator is an abstract class and this function should not be called.")
    
    def remove_file(self):
        os.remove(os.path.join(self.path_dataset, self.name + '.pkl'))
    
    def create_dataset(self):
        l_data = []
        for _ in tqdm.tqdm(range(self.num_examples)):
            example = self.compute_example()
            l_data.append(example)
        return l_data

    def __getitem__(self, i):
        """ Fetch sample at index i """
        return self.data[i]

    def __len__(self):
        """ Get dataset length """
        return len(self.data)