import dgl
import torch

def edge_tensor_to_features(graph: dgl.DGLGraph, features: torch.Tensor, device='cpu'):
    n_edges = graph.number_of_edges()
    resqueeze = False
    if len(features.shape)==3:
        resqueeze=True
        features = features.unsqueeze(-1)
    bs,N,_,n_features = features.shape
    
    ix,iy = graph.edges()
    bsx,bsy = ix//N,iy//N
    Nx,Ny = ix%N,iy%N
    assert torch.all(bsx==bsy), "Edges between graphs, should not be allowed !" #Sanity check
    final_features = features[(bsx,Nx,Ny)] #Here, shape will be (n_edges,n_features)
    if resqueeze:
        final_features = final_features.squeeze(-1)
    return final_features