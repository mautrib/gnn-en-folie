import torch

def mcp_compute_accuracy(raw_scores,target):
    """
    weights and target should be (bs,n,n)
    """
    clique_sizes,_ = torch.max(target.sum(dim=-1),dim=-1) #The '+1' is because the diagonal of the target is 0
    clique_sizes += 1
    bs,n,_ = raw_scores.shape
    true_pos = 0
    total_n_vertices = 0

    probas = torch.sigmoid(raw_scores)

    deg = torch.sum(probas, dim=-1)
    inds = [ (torch.topk(deg[k],int(clique_sizes[k].item()),dim=-1))[1] for k in range(bs)]
    for i,_ in enumerate(raw_scores):
        sol = torch.sum(target[i],dim=1) #Sum over rows !
        ind = inds[i]
        for idx in ind:
            idx = idx.item()
            if sol[idx]:
                true_pos += 1
        total_n_vertices+=clique_sizes[i].item()
    return {'accuracy': true_pos/total_n_vertices}