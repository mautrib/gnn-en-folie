import torch

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