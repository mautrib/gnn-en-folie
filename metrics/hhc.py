import torch

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