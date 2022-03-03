import torch
from toolbox.conversions import dgl_dense_adjacency, edge_format_to_dense_tensor

def f1_score(preds,labels):
    """
    take 2 adjacency matrices and compute precision, recall, f1_score for a tour
    """

    labels = labels.type_as(preds)
    bs, n_nodes ,_  = labels.shape
    true_pos = 0
    false_pos = 0
    mask = torch.ones((n_nodes,n_nodes))-torch.eye(n_nodes)
    mask = mask.type_as(preds)
    for i in range(bs):
        true_pos += torch.sum(mask*preds[i,:,:]*labels[i,:,:]).cpu().item()
        false_pos += torch.sum(mask*preds[i,:,:]*(1-labels[i,:,:])).cpu().item()
        #pos += np.sum(preds[i][0,:] == labels[i][0,:])
        #pos += np.sum(preds[i][1,:] == labels[i][1,:])
    #prec = pos/2*n
    prec = true_pos/(true_pos+false_pos)
    rec = true_pos/(2*n_nodes*bs)
    if prec+rec == 0:
        f1 = 0.0
    else:
        f1 = 2*prec*rec/(prec+rec)
    return prec, rec, f1#, n, bs

def tsp_fgnn_edge_compute_f1(raw_scores,target,k_best=3):
    """
    Computes F1-score with the k_best best edges per row
    For TSP with the chosen 3 best, the best result will be : prec=2/3, rec=1, f1=0.8 (only 2 edges are valid)
    """
    _, ind = torch.topk(raw_scores, k_best, dim = 2) #Here chooses the 3 best choices
    y_onehot = torch.zeros_like(raw_scores)
    y_onehot = y_onehot.type_as(raw_scores)
    y_onehot.scatter_(2, ind, 1)
    prec, rec, f1 =  f1_score(y_onehot,target)
    return {'precision': prec, 'recall': rec, 'f1': f1}

def tsp_dgl_edge_compute_f1(raw_scores, target, k_best=3):
    """
    Computes F1-score with the k_best best edges per row
    For TSP with the chosen 3 best, the best result will be : prec=2/3, rec=1, f1=0.8 (only 2 edges are valid)
     - raw_scores : shape (N_edges, 2)
     - target     : graph with graph.edata['solution'] of shape (N_edges, 1)
    """
    assert raw_scores.shape[1]==2, f"Scores given by model are not given with two classes but with {raw_scores.shape[1]}"
    data = target
    target = target.edata['solution'].squeeze(-1).type_as(raw_scores)
    dense_target = edge_format_to_dense_tensor(target, data)
    proba = torch.softmax(raw_scores,dim=-1)
    proba_of_being_1 = proba[:,1]
    dense_proba_of_1 = edge_format_to_dense_tensor(proba_of_being_1, data)
    N,_ = dense_target.shape
    mask = (dgl_dense_adjacency(data)*(1-torch.eye(N))).type_as(raw_scores)

    _, ind = torch.topk(dense_proba_of_1, k=k_best, dim = 1) #Here chooses the 3 best choices
    y_onehot = torch.zeros_like(dense_target)
    y_onehot = y_onehot.type_as(dense_target)
    y_onehot.scatter_(1, ind, 1)

    true_pos = torch.sum(mask*y_onehot*dense_target).cpu().item()
    false_pos= torch.sum(mask*y_onehot*(1-dense_target)).cpu().item()

    prec = true_pos/(true_pos+false_pos)
    rec = true_pos/(torch.sum(dense_target).cpu().item())
    if prec+rec == 0:
        f1 = 0.0
    else:
        f1 = 2*prec*rec/(prec+rec)
    return {'precision': prec, 'recall': rec, 'f1': f1}

def tsp_mt_edge_compute_f1(raw_scores, target, k_best=3):
    """
    Computes F1-score with the k_best best edges per row
    For TSP with the chosen 3 best, the best result will be : prec=2/3, rec=1, f1=0.8 (only 2 edges are valid)
    """
    bs = len(raw_scores)
    raw_score_tensors = [raw_scores[i] for i in range(bs)]
    target_tensors = [target[i] for i in range(bs)]

    
    true_pos = 0
    false_pos = 0
    total_true_edges = 0
    for raw_score, labels in zip(raw_score_tensors,target_tensors):
        _, ind = torch.topk(raw_score, k_best, dim = 1) #Here chooses the 3 best choices
        y_onehot = torch.zeros_like(raw_score)
        y_onehot = y_onehot.type_as(raw_score)
        y_onehot.scatter_(1, ind, 1)

        n_nodes ,_  = raw_score.shape 
        mask = torch.ones((n_nodes,n_nodes))-torch.eye(n_nodes)
        mask = mask.type_as(raw_score)

        true_pos+= torch.sum(mask*y_onehot*labels).cpu().item()
        false_pos += torch.sum(mask*y_onehot*(1-labels)).cpu().item()
        total_true_edges += torch.sum(labels)

    prec = true_pos/(true_pos+false_pos)
    rec = true_pos/(total_true_edges)
    if prec+rec == 0:
        f1 = 0.0
    else:
        f1 = 2*prec*rec/(prec+rec)

    return {'precision': prec, 'recall': rec, 'f1': f1}
