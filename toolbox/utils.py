import torch
import os
import inspect

# create directory if it does not exist
def check_dir(dir_path):
    dir_path = dir_path.replace('//','/')
    os.makedirs(dir_path, exist_ok=True)

def is_adj(matrix):
    is_shaped = matrix.dim()==2 and matrix.shape[0]==matrix.shape[1]
    is_bool = torch.all((matrix==0) + (matrix==1))
    return is_shaped and is_bool

def restrict_dict_to_function(f, dictionary):
    keys_to_keep = inspect.signature(f).parameters
    d_clean = {}
    for key, value in dictionary.items():
        if key in keys_to_keep:
            d_clean[key] = value
    return d_clean
