from re import A
import torch
import dgl
from toolbox.conversions import dgl_dense_adjacency, edge_format_to_dense_tensor
from toolbox.searches.mcp import mcp_beam_method
from numpy import argwhere as npargwhere
from metrics.common import edgefeat_compute_accuracy

def mcp_fgnn_edge_compute_accuracy(raw_scores,target):
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

def mcp_dgl_edge_compute_f1(raw_scores, target, threshold=0.5):
    """
     - raw_scores : shape (N_edges, 2)
     - target : dgl graph with target.edata['solution'] of shape (N_edges,1)
    """
    assert raw_scores.shape[1]==2, f"Scores given by model are not given with two classes but with {raw_scores.shape[1]}"
    solution = target.edata['solution'].squeeze()
    n_solution_edges = torch.sum(solution)
    #proba = torch.softmax(raw_scores,dim=-1)
    scores_of_being_1 = raw_scores[:,1]
    #dense_proba_of_1 = edge_format_to_dense_tensor(proba_of_being_1, target)
    
    _, ind = torch.topk(scores_of_being_1, k=n_solution_edges)
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

def mcp_dgl_edge_compute_beamsearch_accuracy(raw_scores,target):
    """
     - raw_scores : shape (N_edges, 2)
     - target : dgl graph with target.edata['solution'] of shape (N_edges,1)
    """
    assert raw_scores.shape[1]==2, f"Scores given by model are not given with two classes but with {raw_scores.shape[1]}"
    data = target
    proba = torch.softmax(raw_scores,dim=-1)
    proba_of_being_1 = proba[:,1]
    #dense_proba_of_1 = edge_format_to_dense_tensor(proba_of_being_1, target)

    data.edata['inferred'] = proba_of_being_1
    unbatched_graphs = dgl.unbatch(data)
    adjs = [dgl_dense_adjacency(graph) for graph in unbatched_graphs]
    scores = [graph.edata['inferred'] for graph in unbatched_graphs]
    scores_dense = [edge_format_to_dense_tensor(score, graph) for score,graph in zip(scores,unbatched_graphs)]
    bs = len(scores_dense)
    N,_ = scores_dense[0].shape
    batched_score_dense = torch.zeros((bs, N,N))
    for i,elt in enumerate(scores_dense):
        batched_score_dense[i,:,:] = elt[:,:]

    l_cliques = mcp_beam_method(adjs, batched_score_dense, normalize=False)

    true_pos = 0
    total_count = 0
    for elt, graph_elt in zip(l_cliques, unbatched_graphs):
        target_edge_features = graph_elt.edata['solution']
        target_clique_size = torch.sum(target_edge_features)
        indices = npargwhere((target_edge_features==1).numpy())[:,0]
        assert len(indices)==target_clique_size, "Different target clique size and number of indices of edges. The target matrix doesn't have only zeros and ones ?"
        assert len(indices)>0, "No solution edges ?"
        ins,outs = graph_elt.find_edges(indices)
        if len(elt)>=target_clique_size:
            true_pos += len(indices)
        else:
            for u,v in zip(ins,outs):
                if u in elt and v in elt:
                    true_pos+=1
        total_count += len(indices)

        # for u,v in zip(in_edges, out_edges):
        #     if u in elt and v in elt:
        #         true_pos += 1
        # total = target_elt.n_edges()

    acc = true_pos/total_count
    assert acc<=1, "Accuracy over 1, not normal."
    return {'accuracy': acc}