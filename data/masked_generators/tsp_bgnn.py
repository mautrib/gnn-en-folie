import os
from pkgutil import get_data
import torch
import dgl
import numpy as np
import requests
import zipfile
from scipy.spatial.distance import pdist, squareform
from data.tsp import distance_matrix_tensor_representation
import tqdm
from toolbox import utils

class TSP_BGNN_Generator(torch.utils.data.Dataset):
    def __init__(self, name, args, coeff=1e8):
        self.name=name
        path_dataset = os.path.join(args['path_dataset'], 'tsp_bgnn')
        self.path_dataset = path_dataset
        self.data = []
        
        utils.check_dir(self.path_dataset)#utils.check_dir(self.path_dataset)
        self.constant_n_vertices = False
        self.coeff = coeff
        self.positions = []
        self.filename = os.path.join(self.path_dataset, 'TSP/',f'tsp50-500_{self.name}.txt')

        self.num_neighbors = 25

    def download_files(self):
        basefilepath = os.path.join(self.path_dataset,'TSP.zip')
        print('Downloading Benchmarking GNNs TSP data...')
        url = 'https://www.dropbox.com/s/1wf6zn5nq7qjg0e/TSP.zip?dl=1'
        r = requests.get(url)
        with open(basefilepath,'wb') as f:
            f.write(r.content)
        with zipfile.ZipFile(basefilepath, 'r') as zip_ref:
            zip_ref.extractall(self.path_dataset)

    def load_dataset(self, use_dgl=False):
        """
        Look for required dataset in files and create it if
        it does not exist
        """
        filename = self.name + '.pkl'
        filename_dgl = self.name + '_dgl.pkl'
        path = os.path.join(self.path_dataset, filename)
        path_dgl = os.path.join(self.path_dataset, filename_dgl)
        data_exists = os.path.exists(path)
        data_dgl_exists = os.path.exists(path_dgl)
        if use_dgl and data_dgl_exists:
            print('Reading dataset at {}'.format(path_dgl))
            l_data,l_pos = torch.load(path_dgl)
        elif not use_dgl and data_exists:
            print('Reading dataset at {}'.format(path))
            l_data,l_pos = torch.load(path)
        elif use_dgl:
            print('Reading dataset from BGNN files.')
            l_data,l_pos = self.get_data_from_file(use_dgl=use_dgl)
            print('Saving dataset at {}'.format(path_dgl))
            torch.save((l_data, l_pos), path_dgl)
        else:
            print('Reading dataset from BGNN files.')
            l_data,l_pos = self.get_data_from_file(use_dgl=use_dgl)
            print('Saving dataset at {}'.format(path))
            torch.save((l_data, self.positions), path)
        self.data = list(l_data)
        self.positions = list(l_pos)

    def get_data_from_file(self, use_dgl=False):
        if not os.path.isfile(self.filename):
            self.download_files()

        with open(self.filename, 'r') as f:
            file_data = f.readlines()
        
        l_data,l_pos = [],[]
        print("Processing data...")
        for line in tqdm.tqdm(file_data):
            line = line.split(" ")  # Split into list
            num_nodes = int(line.index('output')//2)
            
            # Convert node coordinates to required format
            nodes_coord = []
            xs,ys = [],[]
            for idx in range(0, 2 * num_nodes, 2):
                x,y = float(line[idx]), float(line[idx + 1])
                xs.append(x)
                ys.append(y)
                nodes_coord.append([float(line[idx]), float(line[idx + 1])])

            # Compute distance matrix
            W_val = squareform(pdist(nodes_coord, metric='euclidean'))
            # Determine k-nearest neighbors for each node
            knns = np.argpartition(W_val, kth=self.num_neighbors, axis=-1)[:, self.num_neighbors::-1]

            # Convert tour nodes to required format
            # Don't add final connection for tour/cycle
            tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]

            # Compute an edge adjacency matrix representation of tour
            edges_target = np.zeros((num_nodes, num_nodes))
            for idx in range(len(tour_nodes) - 1):
                i = tour_nodes[idx]
                j = tour_nodes[idx + 1]
                edges_target[i][j] = 1
                edges_target[j][i] = 1
            # Add final connection of tour in edge target
            edges_target[j][tour_nodes[0]] = 1
            edges_target[tour_nodes[0]][j] = 1

            if use_dgl:
                g = dgl.DGLGraph()
                g.add_nodes(num_nodes)
                g.ndata['feat'] = torch.Tensor(nodes_coord)
                
                edge_feats = []  # edge features i.e. euclidean distances between nodes
                edge_labels = []  # edges_targets as a list
                # Important!: order of edge_labels must be the same as the order of edges in DGLGraph g
                # We ensure this by adding them together
                for idx in range(num_nodes):
                    for n_idx in knns[idx]:
                        if n_idx != idx:  # No self-connection
                            g.add_edge(idx, n_idx)
                            edge_feats.append(W_val[idx][n_idx])
                            edge_labels.append(int(edges_target[idx][n_idx]))
                # dgl.transform.remove_self_loop(g)
                
                # Sanity check
                assert len(edge_feats) == g.number_of_edges() == len(edge_labels)
                
                # Add edge features
                g.edata['feat'] = torch.Tensor(edge_feats).unsqueeze(-1)
                num_nodes = g.num_nodes()
                target_dgl = dgl.graph(g.edges(), num_nodes=num_nodes)
                edge_labels = torch.tensor(edge_labels)
                target_dgl.edata['solution'] = edge_labels
                
                l_data.append((g, target_dgl))
            else:
                W = torch.tensor(W_val,dtype=torch.float)
                B = distance_matrix_tensor_representation(W)

                SOL = torch.zeros((num_nodes,num_nodes),dtype=int)
                prec = tour_nodes[-1]
                for i in range(num_nodes):
                    curr = tour_nodes[i]
                    SOL[curr,prec] = 1
                    SOL[prec,curr] = 1
                    prec = curr
            
                l_data.append((B, SOL))
            l_pos.append((xs,ys))
        return l_data, l_pos
        
    def __getitem__(self, i):
        """ Fetch sample at index i """
        return self.data[i]

    def __len__(self):
        """ Get dataset length """
        return len(self.data)

