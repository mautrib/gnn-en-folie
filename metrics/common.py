import torch
import sklearn.metrics as sk_metrics

### EDGE FEAT

def edgefeat_compute_accuracy(l_inferred, l_targets) -> dict:
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
    return {'accuracy': float(acc)}

def edgefeat_compute_f1(l_inferred, l_targets) -> dict:
    """
     - raw_scores : list of tensor of shape (N_edges_i)
     - target : list of tensors of shape (N_edges_i) (For DGL, from target.edata['solution'], for FGNN, converted)
    """
    assert len(l_inferred)==len(l_targets), f"Size of inferred and target different : {len(l_inferred)} and {len(l_targets)}."
    bs = len(l_inferred)
    prec, rec = 0, 0
    for inferred,solution in zip(l_inferred, l_targets):
        n_solution_edges = torch.sum(solution)
        _, ind = torch.topk(inferred, k=n_solution_edges) 
        y_onehot = torch.zeros_like(inferred)
        y_onehot = y_onehot.type_as(solution)
        y_onehot.scatter_(0, ind, 1)

        prec += sk_metrics.precision_score(solution.detach().cpu().numpy(), y_onehot.detach().cpu().numpy())
        rec  += sk_metrics.recall_score(solution.detach().cpu().numpy(), y_onehot.detach().cpu().numpy())
    prec = prec/bs
    rec = rec/bs
    f1 = 0
    if prec+rec!=0:
        f1 = 2*(prec*rec)/(prec+rec)
    return {'precision': float(prec), 'recall': float(rec), 'f1': float(f1)}

def edgefeat_AUC(l_inferred, l_targets) -> dict:
    """
     - inferred : list of tensor of shape (N_edges_i)
     - target : list of tensors of shape (N_edges_i) (For DGL, from target.edata['solution'], for FGNN, converted)
    """
    assert len(l_inferred)==len(l_targets), f"Size of inferred and target different : {len(l_inferred)} and {len(l_targets)}."
    bs = len(l_inferred)
    auc = 0
    for inferred, target in zip(l_inferred, l_targets):
        auc += sk_metrics.roc_auc_score(target.detach().cpu().numpy(), inferred.detach().cpu().numpy())
    auc = auc/bs
    return {'auc': float(auc)}

def edgefeat_total(l_inferred, l_targets) -> dict:
    final_dict = {}
    final_dict.update(edgefeat_AUC(l_inferred, l_targets))
    final_dict.update(edgefeat_compute_accuracy(l_inferred, l_targets))
    final_dict.update(edgefeat_compute_f1(l_inferred, l_targets))
    return final_dict

### FULL EDGE

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
    return {'accuracy': float(acc)}

def fulledge_compute_f1(l_inferred, l_targets):
    """
     - raw_scores : list of tensors of shape (N,N)
     - target : list of tensors of shape (N,N) (For DGL, for FGNN, the target, for DGL, converted)
    """
    assert len(l_inferred)==len(l_targets), f"Size of inferred and target different : {l_inferred.shape} and {len(l_targets.shape)}."
    bs = len(l_inferred)
    prec, rec = 0, 0
    for cur_inferred, cur_target in zip(l_inferred, l_targets):
        cur_inferred = cur_inferred.flatten()
        cur_target = cur_target.flatten()
        n_solution_edges = cur_target.sum()
        _,ind = torch.topk(cur_inferred, k=n_solution_edges)
        y_onehot = torch.zeros_like(cur_inferred)
        y_onehot = y_onehot.type_as(cur_target)
        y_onehot.scatter_(0, ind, 1)

        prec += sk_metrics.precision_score(cur_target.detach().cpu().numpy(), y_onehot.detach().cpu().numpy())
        rec  += sk_metrics.recall_score(cur_target.detach().cpu().numpy(), y_onehot.detach().cpu().numpy())
    prec = prec/bs
    rec = rec/bs
    f1 = 0
    if prec+rec!=0:
        f1 = 2*(prec*rec)/(prec+rec)
    return {'precision': float(prec), 'recall': float(rec), 'f1': float(f1)}
        
def fulledge_AUC(l_inferred, l_targets) -> dict:
    """
     - l_inferred : list of tensor of shape (N_i, N_i)
     - l_targets : list of tensors of shape (N__i, N_i) (For DGL, from target.edata['solution'], for FGNN, converted)
    """
    assert len(l_inferred)==len(l_targets), f"Size of inferred and target different : {len(l_inferred)} and {len(l_targets)}."
    bs = len(l_inferred)
    auc = 0
    for inferred, target in zip(l_inferred, l_targets):
        auc += sk_metrics.roc_auc_score(target.detach().cpu().numpy().flatten(), inferred.detach().cpu().to(int).numpy().flatten())
    auc = auc/bs
    return {'auc': float(auc)}

def fulledge_total(l_inferred, l_targets) -> dict:
    final_dict = {}
    final_dict.update(fulledge_AUC(l_inferred, l_targets))
    final_dict.update(fulledge_compute_accuracy(l_inferred, l_targets))
    final_dict.update(fulledge_compute_f1(l_inferred, l_targets))
    return final_dict

### NODE

def node_compute_accuracy(l_inferred, l_targets) -> dict:
    """
     - l_inferred : list of tensors of shape (N_nodes_i)
     - l_targets  : list of tensors of shape (N_nodes_i)
    """
    return edgefeat_compute_accuracy(l_inferred, l_targets)

def node_compute_f1(l_inferred, l_targets) -> dict:
    """
     - l_inferred : list of tensors of shape (N_nodes_i)
     - l_targets  : list of tensors of shape (N_nodes_i)
    """
    return edgefeat_compute_f1(l_inferred, l_targets)

def node_AUC(l_inferred, l_targets) -> dict:
    """
     - l_inferred : list of tensors of shape (N_nodes_i)
     - l_targets  : list of tensors of shape (N_nodes_i)
    """
    return edgefeat_AUC(l_inferred, l_targets)

def node_total(l_inferred, l_targets) -> dict:
    final_dict = {}
    final_dict.update(node_AUC(l_inferred, l_targets))
    final_dict.update(node_compute_accuracy(l_inferred, l_targets))
    final_dict.update(node_compute_f1(l_inferred, l_targets))
    return final_dict




