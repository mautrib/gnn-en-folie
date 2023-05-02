import torch
import dgl
import numpy as np
from toolbox.conversions import dgl_dense_adjacency, edge_format_to_dense_tensor
from toolbox.searches.mcp import mcp_beam_method, mcp_beam_method_node
from metrics.common import edgefeat_total as common_edgefeat_total, fulledge_total as common_fulledge_total, node_total as common_node_total

###EDGEFEAT
def edgefeat_beamsearch(l_inferred, l_targets, l_adjacency, beam_size=1280, suffix='') -> dict:
    """
     - l_inferred : list of tensor of shape (N_edges_i)
     - l_targets : list of tensors of shape (N_edges_i) (For DGL, from target.edata['solution'], for FGNN, converted)
     - l_adjacency: list of couples of tensors of size ((N_edges_i), (N_edges_i)) corresponding to the edges (src, dst) of the graph
    """
    raise NotImplementedError("This function is not implemented yet.")

def edgefeat_total(l_inferred, l_targets, l_adjacency) -> dict:
    raise NotImplementedError("This function is not implemented yet.")

###FULLEDGE
def fulledge_beamsearch(l_inferred, l_targets, l_adjacency, beam_size=1280, suffix='') -> dict:
    """
     - l_inferred : list of tensor of shape (N_i,N_i)
     - l_targets : list of tensors of shape (N_i,N_i)
     - l_adjacency: list of adjacency matrices of size (N_i,N_i)
    """
    assert len(l_inferred)==len(l_targets)==len(l_adjacency), f"Size of inferred, target and ajacency different : {len(l_inferred)}, {len(l_targets)} and {len(l_adjacency)}."
    
    raise NotImplementedError("This function is not implemented yet.")

def fulledge_total(l_inferred, l_targets, l_adjacency) -> dict:
    
    raise NotImplementedError("This function is not implemented yet.")

## NODE

def node_beamsearch(l_inferred, l_targets, l_adjacency, beam_size=1280, suffix='') -> dict:
    """
     - l_inferred : list of tensor of shape (N_nodes_i)
     - l_targets : list of tensors of shape (N_nodes_i) (For DGL, from target.ndata['solution'], for FGNN, converted)
     - l_adjacency: list of couples of tensors of size ((N_edges_i), (N_edges_i)) corresponding to the edges (src, dst) of the graph
    """
    assert len(l_inferred)==len(l_targets)==len(l_adjacency), f"Size of inferred, target and adjacency different : {len(l_inferred)}, {len(l_targets)} and {len(l_adjacency)}."
    l_t = [torch.zeros(N,N) for N in [len(elt) for elt in  l_inferred]]
    for t,(src,dst) in zip(l_t, l_adjacency):
        t[src,dst] = 1
        t[dst,src] = 1
    l_adjacency_inverted = [torch.where(t==0) for t in l_t]
    l_src = [[src for src,_ in inverted_edges] for inverted_edges in l_adjacency_inverted]
    l_dst = [[dst for _,dst in inverted_edges] for inverted_edges in l_adjacency_inverted]
    l_adjacency_out = [(src, dst) for (src, dst) in zip(l_src, l_dst)]
    return node_beamsearch(l_inferred, l_targets, l_adjacency_out, beam_size=beam_size, suffix=suffix)
    

def node_total(l_inferred, l_targets, l_adjacency) -> dict:
    final_dict = {}
    final_dict.update(common_node_total(l_inferred, l_targets))
    beam_sizes = [1]
    for beam_size in beam_sizes:
        final_dict.update(node_beamsearch(l_inferred, l_targets, l_adjacency, beam_size=beam_size, suffix=str(beam_size)))
    return final_dict