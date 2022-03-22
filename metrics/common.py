import torch
import dgl

def _edgefeat_compute_accuracy(l_inferred, l_targets):
    """
     - raw_scores : list of tensor of shape (N_edges_i)
     - target : list of tensors of shape (N_edges_i) (For DGL, from target.edata['solution'], for FGNN, converted)
    """
    assert len(l_inferred)==len(l_targets), f"Size of inferred and target different : {len(l_inferred)} and {len(l_targets)}."
    bs = len(l_inferred)
    acc = 0
    for inferred,solution in zip(l_inferred, l_targets):
        n_solution_edges = torch.sum(solution)
        _, ind = torch.topk(inferred, k=n_solution_edges) 
        y_onehot = torch.zeros_like(inferred)
        y_onehot = y_onehot.type_as(solution)
        y_onehot.scatter_(0, ind, 1)

        true_pos = torch.sum(y_onehot*solution)
        acc += true_pos/n_solution_edges
    acc = acc/bs
    assert acc<=1, "Accuracy over 1, not normal."
    return {'accuracy':acc}

def edgefeat_compute_accuracy(raw_scores, target, data=None):
    if isinstance(target, dgl.DGLGraph):
        proba = torch.softmax(raw_scores,dim=-1)
        proba_of_being_1 = proba[:,1]
        
        target.edata['inferred'] = proba_of_being_1
        unbatched_graphs = dgl.unbatch(target)
        l_rs = [graph.edata['inferred'] for graph in unbatched_graphs]
        l_target = [graph.edata['solution'].squeeze() for graph in unbatched_graphs]
    else:
        assert data is not None, "No data, can't find adjacency"
        assert data.ndim==4, "Data not recognized"
        adjacency = data[:,:,:,1]
        l_srcdst = [(torch.where(adj>0)) for adj in adjacency]
        l_rs = [ graph[src,dst] for (graph,(src,dst)) in zip(raw_scores,l_srcdst)]
        l_target = [ graph[src,dst] for (graph,(src,dst)) in zip(target,l_srcdst)]
    return _edgefeat_compute_accuracy(l_rs, l_target)