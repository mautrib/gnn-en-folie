import torch
import os
import inspect
import copy

# create directory if it does not exist
def check_dir(dir_path):
    dir_path = dir_path.replace('//','/')
    os.makedirs(dir_path, exist_ok=True)

def is_adj(matrix):
    is_shaped = matrix.dim()==2 and matrix.shape[0]==matrix.shape[1]
    is_bool = torch.all((matrix==0) + (matrix==1))
    return is_shaped and is_bool

def restrict_dict_to_function(f, dictionary):
    """Restriction of the the dict 'dictionary' to fit to the arguments of func 'f'"""
    keys_to_keep = inspect.signature(f).parameters
    d_clean = {}
    for key, value in dictionary.items():
        if key in keys_to_keep:
            d_clean[key] = value
    return d_clean

def get_accelerator_dict(device: str):
    """This function will cut down the device input to something the pl.Trainer can read"""
    accelerator_dict = {}
    
    accelerator, number_of_devices = device, 1
    if ':' in device:
        accelerator, number_of_devices = device.split(':')
        number_of_devices = int(number_of_devices)

    accelerator_dict['accelerator'] = accelerator

    if accelerator=='cpu':
        accelerator_dict['num_processes'] = number_of_devices
    elif accelerator=='gpu':
        accelerator_dict['gpus'] = number_of_devices
    elif accelerator=='tpu':
        accelerator_dict['tpu_cores'] = number_of_devices
    elif accelerator=='ipu':
        accelerator_dict['ipus'] = number_of_devices
    elif accelerator=='auto':
        pass
    else:
        raise NotImplementedError(f"Accelerator {accelerator} not recognized.")
    return accelerator_dict

def clean_config(config):
    problem_key = config['problem']
    arch_name = config['arch']['name']
    clean_one = copy.deepcopy(config)
    for key in config['data']['train']['problems']: #Other data values for train/val
        if key!=problem_key:
            del clean_one['data']['train']['problems'][key]
    for key in config['data']['test']['problems']:  #Other data values for test
        if key!=problem_key:
            del clean_one['data']['test']['problems'][key]
    for key in config['arch']['configs']:  #Other architectures
        if key!=arch_name:
            del clean_one['arch']['configs'][key]
    return clean_one

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']