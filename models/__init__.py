from models.dgl_edge import DGL_Edge
from models.dgl_node import DGL_Node
from models.fgnn_edge import FGNN_Edge
from models.fgnn_node import FGNN_Node
from models.dgl.gatedgcn import GatedGCNNet_Edge
from models.fgnn.fgnn import Simple_Edge_Embedding, Simple_Node_Embedding
import logging

FGNN_EMBEDDING_DICT = {
    'edge': FGNN_Edge,
    'node': FGNN_Node
}

DGL_EMBEDDING_DICT = {
    'edge': DGL_Edge,
    'node': DGL_Node
}

MODULE_DICT = {
    'fgnn' : {  'edge': Simple_Edge_Embedding,
                'node': Simple_Node_Embedding   },
    'gatedgcn' : {  'edge': GatedGCNNet_Edge    }
}

NOT_DGL_ARCHS = ('fgnn',)

def check_dgl_compatibility(use_dgl, arch_name, dgl_check=True):
    arch_uses_dgl = not(arch_name in NOT_DGL_ARCHS)
    warning_str=''
    if use_dgl and not(arch_uses_dgl):
        warning_str = f"Architecture '{arch_name}' is registered as not using DGL but you want it to. If it should use DGL, please remove '{arch_name}' from variable 'NOT_DGL_ARCHS' in models/__init__.py"
    elif not(use_dgl) and arch_uses_dgl:
        warning_str = f"Architecture '{arch_name}' is registered as using DGL but you're not using DGL. If it shouldn't use DGL, please add '{arch_name}' it to variable 'NOT_DGL_ARCHS' in models/__init__.py"
    if warning_str:
        if dgl_check:
            raise TypeError(warning_str)
        else:
            logging.exception(warning_str)

def get_pipeline(config, dgl_check=True):
    arch_name = config['arch']['name'].lower()
    embedding = config['arch']['embedding'].lower()
    use_dgl = config['arch']['use_dgl']
    check_dgl_compatibility(use_dgl, arch_name, dgl_check=dgl_check)
    if use_dgl:
        PL_Model = DGL_EMBEDDING_DICT[embedding]
    else:
        PL_Model = FGNN_EMBEDDING_DICT[embedding]
    
    Module_Class = MODULE_DICT[arch_name][embedding]
    module_config = config['arch']['configs'][arch_name]
    module = Module_Class(**module_config)

    optim_config = config['train']['optim_args']
    pipeline = PL_Model(module, optim_config)

    return pipeline



