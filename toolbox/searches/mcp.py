import torch

def insert(container, new_item, key=len):
    """
    Just a dichotomy to place an item into a tensor depending on a key, supposing the list is ordered in a decresc manner
    """
    if len(container)==0:
        return [new_item]

    l,r = 0, len(container)
    item_value = key(new_item)
    while l!=r:
        mid = (l+r)//2
        if key(container[mid])>=item_value:
            l=mid+1
        else:
            r = mid
    return container[:l] + [new_item] + container[l:]

def mcp_beam_method(adjs, raw_scores, seeds=None, add_singles=True, beam_size=1280):
    """
    The idea of this method is to establish a growing clique, keeping only the biggest cliques starting from the most probable nodes
    seeds should be a list of sets
    """
    seeding = (seeds is not None)

    solo=False
    if len(raw_scores.shape)==2:
        solo=True
        raw_scores = raw_scores.unsqueeze(0)
        adjs = adjs.unsqueeze(0)
        if seeding: seeds = [seeds] #In that case we'd only have a set
    

    bs,n,_ = raw_scores.shape

    probas = torch.sigmoid(raw_scores)

    degrees = torch.sum(probas, dim=-1)
    inds_order = torch.argsort(degrees,dim=-1,descending=True) #Sort them in ascending order
    
    l_clique_inf = []
    for k in range(bs): #For the different data in the batch
        cliques = [] #Will contain 1D Tensors
        cur_adj = adjs[k]
        node_order = torch.arange(n)[inds_order[k]] #Creates the node order
        if seeding:
            seed = seeds[k]
            node_order = [elt.item() for elt in node_order if not elt.item() in seed] #Remove the elements of the seed
            cliques.append(torch.tensor([elt for elt in seed]))
        for cur_step in range(len(node_order)):
            cur_node = node_order[cur_step]
            for clique in cliques: #Iterate over the currently saved cliques to make them grow
                t_clique = clique.clone().detach()
                neighs = cur_adj[cur_node][t_clique]
                if torch.all(neighs==1): #If all clique nodes are adjacent to cur_node
                    new_clique = torch.cat((clique,torch.tensor([cur_node],dtype=torch.long)))
                    cliques = insert(cliques,new_clique)
            if add_singles: cliques = insert(cliques,torch.tensor([cur_node])) #Add the clique with just the node
            cliques = cliques[:beam_size] # Keep a good size
        #Now choose one of the best, knowing cliques is ordered descendingly
        #I just choose the first one, but we can choose the most similar to solution ?
        best_set = set([elt.item() for elt in cliques[0]])
        l_clique_inf.append(best_set)
    if solo:
        l_clique_inf = l_clique_inf[0]
    return l_clique_inf