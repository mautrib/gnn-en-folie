import torch
import os

# create directory if it does not exist
def check_dir(dir_path):
    dir_path = dir_path.replace('//','/')
    os.makedirs(dir_path, exist_ok=True)

def is_adj(matrix):
    is_shaped = matrix.dim()==2 and matrix.shape[0]==matrix.shape[1]
    is_bool = torch.all((matrix==0) + (matrix==1))
    return is_shaped and is_bool