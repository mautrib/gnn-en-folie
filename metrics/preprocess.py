import torch
import dgl
from toolbox.conversions import edge_format_to_dense_tensor

def fulledge_converter(raw_scores, target, data=None, **kwargs):
    if isinstance(target, dgl.DGLGraph):
        proba = torch.softmax(raw_scores,dim=-1)
        proba_of_being_1 = proba[:,1]
        target.edata['inferred'] = proba_of_being_1
        unbatched_graphs = dgl.unbatch(target)
        l_inferred = [edge_format_to_dense_tensor(graph.edata['inferred'], graph) for graph in unbatched_graphs]
        l_targets = [edge_format_to_dense_tensor(graph.edata['solution'], graph).squeeze() for graph in unbatched_graphs]
        l_adjacency = [edge_format_to_dense_tensor(torch.ones(graph.num_edges()),graph) for graph in unbatched_graphs]
    else:
        assert data is not None, "No data, can't find adjacency"
        assert data.ndim==4, "Data not recognized"
        adjacency = data[:,:,:,1]
        l_inferred = [rs for rs in raw_scores]
        l_targets = [t for t in target]
        l_adjacency = [a for a in adjacency]
    return l_inferred, l_targets, l_adjacency

def edgefeat_converter(raw_scores, target, data=None, **kwargs):
    if isinstance(target, dgl.DGLGraph):
        proba = torch.softmax(raw_scores,dim=-1)
        proba_of_being_1 = proba[:,1]
        
        target.edata['inferred'] = proba_of_being_1
        unbatched_graphs = dgl.unbatch(target)
        l_inferred = [graph.edata['inferred'] for graph in unbatched_graphs]
        l_target = [graph.edata['solution'].squeeze() for graph in unbatched_graphs]
        l_adjacency = [graph.edges() for graph in unbatched_graphs]
    else:
        assert data is not None, "No data, can't find adjacency"
        assert data.ndim==4, "Data not recognized"
        adjacency = data[:,:,:,1]
        l_adjacency = [(torch.where(adj>0)) for adj in adjacency]
        l_inferred = [ graph[src,dst] for (graph,(src,dst)) in zip(raw_scores,l_adjacency)]
        l_target = [ graph[src,dst] for (graph,(src,dst)) in zip(target,l_adjacency)]
    return l_inferred, l_target, l_adjacency