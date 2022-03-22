import torch
import dgl
from metrics.common import edgefeat_compute_accuracy

def sbm_fgnn_edge_compute_accuracy(raw_scores,target):
    """
    Computes a simple category accuracy
    Needs raw_scores.shape = (bs,n,n) and target.shape = (bs,n,n)
    """

    bs,n,_ = raw_scores.shape

    _,ind = torch.topk(raw_scores, n//2, -1)
    y_onehot = torch.zeros_like(raw_scores).type_as(raw_scores)
    y_onehot.scatter_(2, ind, 1)

    true_pos = bs * n * n - int(torch.sum(torch.abs(target-y_onehot)))
    acc = true_pos/(bs*n*n)
    return {'accuracy': acc}

def sbm_dgl_edge_compute_f1(raw_scores, target):
    """
     - raw_scores : shape (N_edges, 2)
     - target : dgl graph with target.edata['solution'] of shape (N_edges,1)
    """
    assert raw_scores.shape[1]==2, f"Scores given by model are not given with two classes but with {raw_scores.shape[1]}"
    solution = target.edata['solution'].squeeze()
    n_solution_edges = torch.sum(solution)
    scores_of_being_1 = raw_scores[:,1]
    
    _, ind = torch.topk(scores_of_being_1, k=n_solution_edges) #Here chooses the 3 best choices
    y_onehot = torch.zeros_like(solution)
    y_onehot = y_onehot.type_as(solution)
    y_onehot.scatter_(0, ind, 1)

    true_pos = torch.sum(y_onehot*solution)
    false_pos = torch.sum(y_onehot*(1-solution))
    prec = true_pos/(true_pos+false_pos)
    rec = true_pos/n_solution_edges
    if prec+rec == 0:
        f1 = 0.0
    else:
        f1 = 2*prec*rec/(prec+rec)
    return {'precision': prec, 'recall': rec, 'f1': f1}

def sbm_dgl_edge_compute_accuracy_unbatch(raw_scores, target):
    """
     - raw_scores : shape (N_edges, 2)
     - target : dgl graph with target.edata['solution'] of shape (N_edges,1)
    """
    assert raw_scores.shape[1]==2, f"Scores given by model are not given with two classes but with {raw_scores.shape[1]}"
    proba = torch.softmax(raw_scores,dim=-1)
    proba_of_being_1 = proba[:,1]
    
    target.edata['inferred'] = proba_of_being_1
    unbatched_graphs = dgl.unbatch(target)
    bs = len(unbatched_graphs)
    acc = 0
    for graph in unbatched_graphs:
        solution = graph.edata['solution'].squeeze()
        inferred = graph.edata['inferred']
        n_solution_edges = torch.sum(solution)
        _, ind = torch.topk(inferred, k=n_solution_edges) #Here chooses the 3 best choices
        y_onehot = torch.zeros_like(inferred)
        y_onehot = y_onehot.type_as(solution)
        y_onehot.scatter_(0, ind, 1)

        true_pos = torch.sum(y_onehot*solution)
        acc += true_pos/n_solution_edges
    acc = acc/bs
    assert acc<=1, "Accuracy over 1, not normal."
    return {'accuracy':acc}

def sbm_edgefeat_compute_accuracy(raw_scores, target, data=None):
    if isinstance(target, dgl.DGLGraph):
        proba = torch.softmax(raw_scores,dim=-1)
        proba_of_being_1 = proba[:,1]
        
        target.edata['inferred'] = proba_of_being_1
        unbatched_graphs = dgl.unbatch(target)
        l_rs = [graph.edata['inferred'] for graph in unbatched_graphs]
        l_target = [graph.edata['solution'].squeeze() for graph in unbatched_graphs]
    else:
        assert data is not None, "No data, can't find adjacency"
        assert data.ndim()==4, "Data not recognized"
        adjacency = data[:,:,:,1]
        l_srcdst = [(torch.where(adj>0)) for adj in adjacency]
        l_rs = [ graph[src,dst] for (graph,(src,dst)) in zip(raw_scores,l_srcdst)]
        l_target = [ graph[src,dst] for (graph,(src,dst)) in zip(target,l_srcdst)]
    return edgefeat_compute_accuracy(l_rs, l_target)