import torch
import dgl
from metrics.common import edgefeat_compute_accuracy

def hhc_fgnn_edge_compute_accuracy(raw_scores, target):
    """ Computes simple accuracy by choosing the most probable edge
    For HHC:    - raw_scores and target of shape (bs,n,n)
                - target should be ones over the diagonal for the normal HHC (but can be changed to fit another solution)
     """
    bs,n,_ = raw_scores.shape
    _, ind = torch.topk(raw_scores, 1, dim = 2) #Here chooses the best choice
    y_onehot = torch.zeros_like(raw_scores).type_as(raw_scores)
    y_onehot.scatter_(2, ind, 1)
    accu = target*y_onehot #Places 1 where the values are the same
    true_pos = torch.count_nonzero(accu).item()
    n_total = bs * n #Perfect would be that we have the right permutation for every bs 
    acc = true_pos/n_total
    return {'accuracy': acc}

def hhc_dgl_edge_compute_f1(raw_scores, target):
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

def hhc_dgl_edge_compute_accuracy_unbatch(raw_scores, target):
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