import torch

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