import torch

def edgefeat_compute_accuracy(l_inferred, l_targets):
    """
     - raw_scores : list of tensor of shape (N_edges_i)
     - target : list of tensors of shape (N_edges_i) (For DGL, from target.edata['solution'], for FGNN, converted)
    """
    assert len(l_inferred)==len(l_targets), f"Size of inferred and target different : {len(l_inferred)} and {len(l_targets)}."
    bs = len(l_inferred)
    acc = 0
    for inferred,solution in zip(l_inferred, l_targets):
        n_solution_edges = torch.sum(solution)
        n_edges = len(solution)
        _, ind = torch.topk(inferred, k=n_solution_edges) 
        y_onehot = torch.zeros_like(inferred)
        y_onehot = y_onehot.type_as(solution)
        y_onehot.scatter_(0, ind, 1)

        true_cat = torch.sum(y_onehot*solution) + torch.sum((1-y_onehot)*(1-solution))
        acc += true_cat/n_edges
    acc = acc/bs
    assert acc<=1, "Accuracy over 1, not normal."
    return {'accuracy':acc}

def fulledge_compute_accuracy(l_inferred, l_targets):
    """
     - raw_scores : list of tensors of shape (N,N)
     - target : list of tensors of shape (N,N) (For DGL, for FGNN, the target, for DGL, converted)
    """
    assert len(l_inferred)==len(l_targets), f"Size of inferred and target different : {l_inferred.shape} and {len(l_targets.shape)}."
    bs = len(l_inferred)
    acc=0
    for cur_inferred, cur_target in zip(l_inferred, l_targets):
        cur_inferred = cur_inferred.flatten()
        cur_target = cur_target.flatten()
        n_solution_edges = cur_target.sum()
        n_edges = len(cur_target)
        _,ind = torch.topk(cur_inferred, k=n_solution_edges)
        y_onehot = torch.zeros_like(cur_inferred)
        y_onehot = y_onehot.type_as(cur_target)
        y_onehot.scatter_(0, ind, 1)
        acc += (torch.sum(y_onehot*cur_target) + torch.sum((1-y_onehot)*(1-cur_target)))/(n_edges)
    acc = acc/bs
    assert acc<=1, "Accuracy over 1, not normal."
    return {'accuracy':acc}

