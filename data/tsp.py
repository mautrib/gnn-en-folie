import os
import torch
import dgl
from data.base import Base_Generator, connectivity_to_dgl, dense_tensor_to_edge_format
from toolbox import utils
import math
import random
import networkx
from numpy.random import default_rng
import tqdm


try:
    from concorde.tsp import TSPSolver
except ModuleNotFoundError:
    print("Trying to continue without pyconcorde as it is not installed. TSP data generation will fail.")

rng = default_rng(41)


GENERATOR_FUNCTIONS_TSP = {}

##TSP Generation functions

def dist_from_pos(pos):
    N = len(pos)
    W_dist = torch.zeros((N,N))
    for i in range(0,N-1):
        for j in range(i+1,N):
            curr_dist = math.sqrt( (pos[i][0]-pos[j][0])**2 + (pos[i][1]-pos[j][1])**2)
            W_dist[i,j] = curr_dist
            W_dist[j,i] = curr_dist
    return W_dist

def generates_TSP(name):
    """ Register a generator function for a graph distribution """
    def decorator(func):
        GENERATOR_FUNCTIONS_TSP[name] = func
        return func
    return decorator

@generates_TSP("GaussNormal")
def generate_gauss_normal_netx(N):
    """ Generate random graph with points"""
    pos = {i: (random.gauss(0, 1), random.gauss(0, 1)) for i in range(N)} #Define the positions of the points
    W_dist = dist_from_pos(pos)
    g = networkx.random_geometric_graph(N,0,pos=pos)
    g.add_edges_from(networkx.complete_graph(N).edges)
    return g, torch.as_tensor(W_dist, dtype=torch.float)

@generates_TSP("Square01")
def generate_square_netx(N):
    pos = {i: (random.random(), random.random()) for i in range(N)} #Define the positions of the points
    W_dist = dist_from_pos(pos)
    g = networkx.random_geometric_graph(N,0,pos=pos)
    g.add_edges_from(networkx.complete_graph(N).edges)
    return g, torch.as_tensor(W_dist, dtype=torch.float)

def distance_matrix_tensor_representation(W):
    """ Create a tensor B[:,:,1] = W and B[i,i,0] = deg(i)"""
    W_adjacency = torch.sign(W)
    degrees = W_adjacency.sum(1)
    B = torch.zeros((len(W), len(W), 2))
    B[:, :, 1] = W
    indices = torch.arange(len(W))
    B[indices, indices, 0] = degrees
    return B

# def normalize_tsp(xs,ys):
#     """ 'Normalizes' points positions by moving they in a way where the principal component of the point cloud is directed vertically"""
#     X = [(x,y) for x,y in zip(xs,ys)]
#     pca = PCA(n_components=1)
#     pca.fit(X)
#     pc = pca.components_[0]
#     rot_angle = pi/2 - angle(pc[0]+1j*pc[1])
#     x_rot = [ x*cos(rot_angle) - y*sin(rot_angle) for x,y in X ]
#     y_rot = [ x*sin(rot_angle) + y*cos(rot_angle) for x,y in X ]
#     return x_rot,y_rot

class TSP_Generator(Base_Generator):
    """
    Traveling Salesman Problem Generator.
    Uses the pyconcorde wrapper : see https://github.com/jvkersch/pyconcorde (thanks a lot)
    """
    def __init__(self, name, args, coeff=1e8):
        self.generative_model = args['generative_model']
        self.distance = args['distance_used']
        num_examples = args['num_examples_' + name]
        self.n_vertices = args['n_vertices']
        subfolder_name = 'TSP_{}_{}_{}_{}'.format(self.generative_model, 
                                                     self.distance,
                                                     num_examples,
                                                     self.n_vertices)
        path_dataset = os.path.join(args['path_dataset'], 'tsp', subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        
        
        utils.check_dir(self.path_dataset)#utils.check_dir(self.path_dataset)
        self.constant_n_vertices = True
        self.coeff = coeff
        self.positions = []
    
    def load_dataset(self, use_dgl=False):
        """
        Look for required dataset in files and create it if
        it does not exist
        """
        filename = self.name + '.pkl'
        filename_dgl = self.name + '_dgl.pkl'
        path = os.path.join(self.path_dataset, filename)
        path_dgl = os.path.join(self.path_dataset, filename_dgl)
        if os.path.exists(path):
            if use_dgl:
                print('Reading dataset at {}'.format(path_dgl))
                data,pos = torch.load(path_dgl)
            else:
                print('Reading dataset at {}'.format(path))
                data,pos = torch.load(path)
            print('Reading dataset at {}'.format(path))
            self.data = list(data)
            self.positions = list(pos)
        else:
            print('Creating dataset at {}'.format(path))
            l_data = self.create_dataset()
            print('Saving dataset at {}'.format(path))
            torch.save((l_data, self.positions), path)
            print('Creating dataset at {}'.format(path_dgl))
            print("Converting data to DGL format")
            l_data_dgl = []
            for data,target in tqdm.tqdm(l_data):
                elt_dgl = connectivity_to_dgl(data)
                target_dgl = self._solution_conversion(target, elt_dgl)
                l_data_dgl.append((elt_dgl,target_dgl))
            print("Conversion ended.")
            print('Saving dataset at {}'.format(path_dgl))
            torch.save((l_data_dgl, self.positions), path_dgl)
            if use_dgl:
                self.data = l_data_dgl
            else:
                self.data = l_data

    def compute_example(self):
        """
        Compute pairs (Adjacency, Optimal Tour)
        """
        try:
            g, W = GENERATOR_FUNCTIONS_TSP[self.generative_model](self.n_vertices)
        except KeyError:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        xs = [g.nodes[node]['pos'][0] for node in g.nodes]
        ys = [g.nodes[node]['pos'][1] for node in g.nodes]

        problem = TSPSolver.from_data([self.coeff*elt for elt in xs],[self.coeff*elt for elt in ys],self.distance) #1e8 because Concorde truncates the distance to the nearest integer
        solution = problem.solve(verbose=False)
        assert solution.success, f"Couldn't find solution! \n x =  {xs} \n y = {ys} \n {solution}"

        B = distance_matrix_tensor_representation(W)
        
        SOL = torch.zeros((self.n_vertices,self.n_vertices),dtype=int)
        prec = solution.tour[-1]
        for i in range(self.n_vertices):
            curr = solution.tour[i]
            SOL[curr,prec] = 1
            SOL[prec,curr] = 1
            prec = curr
        
        self.positions.append((xs,ys))
        return (B, SOL)
    
    @staticmethod
    def _solution_conversion(target, dgl_graph):
        num_nodes = dgl_graph.num_nodes()
        target_dgl = dgl.graph(dgl_graph.edges(), num_nodes=num_nodes)
        edge_features = dense_tensor_to_edge_format(target, target_dgl)
        target_dgl.edata['solution'] = edge_features
        return target_dgl