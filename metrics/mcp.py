import torch
import dgl
from toolbox.conversions import dgl_dense_adjacency, edge_format_to_dense_tensor
from toolbox.searches.mcp import mcp_beam_method
from metrics.common import edgefeat_total as common_edgefeat_total, fulledge_total as common_fulledge_total

###EDGEFEAT
def edgefeat_beamsearch(l_inferred, l_targets, l_adjacency) -> dict:
    """
     - l_inferred : list of tensor of shape (N_edges_i)
     - l_targets : list of tensors of shape (N_edges_i) (For DGL, from target.edata['solution'], for FGNN, converted)
     - l_adjacency: list of couples of tensors of size ((N_edges_i), (N_edges_i)) corresponding to the edges (src, dst) of the graph
    """

    l_dgl = [dgl.graph((src,dst)) for src,dst in l_adjacency]
    full_inferred = [edge_format_to_dense_tensor(inferred,graph) for inferred,graph in zip(l_inferred, l_dgl)]
    full_target = [edge_format_to_dense_tensor(target,graph) for target,graph in zip(l_targets, l_dgl)]
    full_adjacency = [dgl_dense_adjacency(graph) for graph in l_dgl]

    return fulledge_beamsearch(full_inferred, full_target, full_adjacency)

def edgefeat_total(l_inferred, l_targets, l_adjacency) -> dict:
    final_dict = {}
    final_dict.update(common_edgefeat_total(l_inferred, l_targets))
    final_dict.update(edgefeat_beamsearch(l_inferred, l_targets, l_adjacency))
    return final_dict

###FULLEDGE
def fulledge_beamsearch(l_inferred, l_targets, l_adjacency) -> dict:
    """
     - l_inferred : list of tensor of shape (N_i,N_i)
     - l_targets : list of tensors of shape (N_i,N_i)
     - l_adjacency: list of adjacency matrices of size (N_i,N_i)
    """
    assert len(l_inferred)==len(l_targets)==len(l_adjacency), f"Size of inferred, target and ajacency different : {len(l_inferred)}, {len(l_targets)} and {len(l_adjacency)}."
    bs = len(l_inferred)

    l_cliques = mcp_beam_method(l_adjacency, l_inferred, normalize=False)

    true_pos = 0
    total_count = 0
    size_error_percentage = 0
    for inferred_clique, target in zip(l_cliques, l_targets):
        target_degrees = target.sum(-1)
        target_clique_set = set(torch.where(target_degrees>0)[0].detach().cpu().numpy())
        target_clique_size = len(target_clique_set)
        inf_clique_size = len(inferred_clique)
        
        true_pos += len(target_clique_set.intersection(inferred_clique))
        total_count += target_clique_size

        size_error_percentage += (inf_clique_size-target_clique_size)/target_clique_size

    size_error_percentage/=bs
    acc = true_pos/total_count
    assert acc<=1, "Accuracy over 1, not normal."
    return {'bs-accuracy': acc, 'bs-size_error_percentage': size_error_percentage}

def fulledge_total(l_inferred, l_targets, l_adjacency) -> dict:
    final_dict = {}
    final_dict.update(common_fulledge_total(l_inferred, l_targets))
    final_dict.update(fulledge_beamsearch(l_inferred, l_targets, l_adjacency))
    return final_dict
