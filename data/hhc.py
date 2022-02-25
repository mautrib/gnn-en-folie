from data.base import Base_Generator
from toolbox import utils
import os
import torch
from numpy.random import default_rng
import dgl
from toolbox.conversions import dense_tensor_to_edge_format

rng = default_rng(41)

GENERATOR_FUNCTIONS_HHC = {}

def generates_HHC(name):
    """Registers a generator function for HHC problem """
    def decorator(func):
        GENERATOR_FUNCTIONS_HHC[name] = func
        return func
    return decorator

@generates_HHC('Gauss')
def generate_gauss_hhc(n,lam,mu):
    """ Using gaussian distribution for the HHC. The information limit is $\mu^2 \ge 4log(n)$ for lambda=0"""
    W_weights = rng.normal(loc=mu,scale=1,size=(n,n)) #Outside of the cycle
    diag = rng.normal(loc=lam,scale=1,size=n) #HHC cycle distribution 
    for i in range(n):
        W_weights[i,(i+1)%n] = diag[i]
    W_weights/=W_weights.std()
    W_weights-=W_weights.mean()
    return W_weights

@generates_HHC('UniformMean')
def generate_uniform(n,lam,mu):
    """ Using uniform distribution for the HHC, with a moved mean """
    W_weights = rng.uniform(size=(n,n))+mu
    diag = rng.uniform(size=n)+lam
    for i in range(n):
        W_weights[i,(i+1)%n] = diag[i]
    W_weights/=W_weights.std()
    W_weights-=W_weights.mean()
    return W_weights

@generates_HHC('Poisson')
def generate_poisson_hhc(n,lam,mu):
    raise NotImplementedError

def weight_matrix_to_tensor_representation(W):
    """ Create a tensor B[:,:,1] = W and B[i,i,0] = deg(i)"""
    N = len(W)
    degrees = N*torch.ones(N).to(torch.float)
    B = torch.zeros((N, N, 2), dtype=torch.float)
    B[:, :, 1] = W
    indices = torch.arange(len(W))
    B[indices, indices, 0] = degrees
    return B

class HHC_Generator(Base_Generator):
    """
    Hidden Hamilton Cycle Generator
    See article : https://arxiv.org/abs/1804.05436
    """
    def __init__(self, name, args):
        self.generative_model = args['generative_model']
        self.cycle_param = args['cycle_param']
        self.fill_param  = args['fill_param']
        num_examples = args['num_examples_' + name]
        self.n_vertices = args['n_vertices']
        subfolder_name = 'HHC_{}_{}_{}_{}_{}'.format(self.generative_model, 
                                                     self.cycle_param,
                                                     self.fill_param,
                                                     num_examples,
                                                     self.n_vertices)
        path_dataset = os.path.join(args['path_dataset'],'hhc', subfolder_name)
        super().__init__(name, path_dataset, num_examples)
        self.data = []
        
        
        utils.check_dir(self.path_dataset)#utils.check_dir(self.path_dataset)
        self.constant_n_vertices = True

    def compute_example(self):
        try:
            W = GENERATOR_FUNCTIONS_HHC[self.generative_model](self.n_vertices,self.cycle_param,self.fill_param)
        except KeyError:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        
        W = torch.tensor(W,dtype=torch.float)
        SOL = torch.eye(self.n_vertices).roll(1,dims=-1)
        SOL = (SOL+SOL.T)
        #W,SOL = utils.permute_adjacency_twin(W,SOL)
        
        B = weight_matrix_to_tensor_representation(W)
        return (B,SOL)
    
    @classmethod
    def _solution_conversion(cls, target, dgl_graph):
        num_nodes = dgl_graph.num_nodes()
        target_dgl = dgl.graph(dgl_graph.edges(), num_nodes=num_nodes)
        edge_classif = dense_tensor_to_edge_format(target, target_dgl)
        target_dgl.edata['solution'] = edge_classif
        return target_dgl