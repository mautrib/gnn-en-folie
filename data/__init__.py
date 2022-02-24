from torch.utils.data import DataLoader
from dgl import batch as dglbatch
from models import check_dgl_compatibility

from data.tsp import TSP_Generator
from data.mcp import MCP_Generator
from data.sbm import SBM_Generator
from data.hhc import HHC_Generator

def get_generator_class(problem_key):
    if problem_key == 'tsp':
        Generator_Class = TSP_Generator
    elif problem_key == 'mcp':
        Generator_Class = MCP_Generator
    elif problem_key == 'sbm':
        Generator_Class = SBM_Generator
    elif problem_key == 'hhc':
        Generator_Class = HHC_Generator
    else:
        raise NotImplementedError(f"Generator for problem {problem_key} hasn't been implemented or define in data/__init__.py yet.")
    return Generator_Class

TRAIN_VAL_TEST_LOOKUP = {
    'train': 'train',
    'val': 'train',
    'test': 'test'
} #If we're validating or testing, we'll check the config under the 'train' key. For testing, it's 'test'

def tensor_to_pytorch(generator, batch_size=32, shuffle=False, num_workers=4, **kwargs):
    pytorch_loader = DataLoader(generator, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers)
    return pytorch_loader

def _collate_fn_dgl(samples_list):
    bs = len(samples_list)
    input1_list = [input1 for (input1, _) in samples_list]
    target_list = [target for (_,target) in samples_list]
    input_batch = dglbatch(input1_list)
    target_batch = dglbatch(target_list)
    return (input_batch,target_batch)

def dgl_to_pytorch(generator, batch_size=32, shuffle=False, num_workers=4, **kwargs):
    pytorch_loader = DataLoader(generator, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, collate_fn=_collate_fn_dgl)
    return pytorch_loader

def get_generator(config:dict, type:str):
    problem_key = config['problem'].lower()
    Generator_Class = get_generator_class(problem_key)
    
    lookup_key = TRAIN_VAL_TEST_LOOKUP[type]

    data_config = config['data'][lookup_key]
    data_config['path_dataset'] = config['data']['path_dataset']
    problem_specific_config = config['data'][lookup_key]['problems'][problem_key]

    dataset = Generator_Class(type, {**data_config, **problem_specific_config})
    return dataset

def get_dataset(config:dict, type:str, dgl_check=True,dataloader_args={}):

    dataset = get_generator(config, type)

    use_dgl = config['arch']['use_dgl']
    arch_name = config['arch']['name'].lower()
    check_dgl_compatibility(use_dgl, arch_name, dgl_check=dgl_check)

    dataset.load_dataset(use_dgl=use_dgl)

    loader_config = config['train']
    if use_dgl:
        dataloaded = dgl_to_pytorch(dataset, **loader_config, **dataloader_args)
    else:
        dataloaded = tensor_to_pytorch(dataset,**loader_config, **dataloader_args)

    return dataloaded

def get_train_val_generators(config:dict):
    train_gen = get_generator(config, 'train')
    val_gen = get_generator(config, 'val')
    return train_gen, val_gen

def get_test_generator(config:dict):
    test_gen = get_generator(config, 'test')
    return test_gen

def get_train_val_datasets(config:dict, dgl_check=True):
    train_dataset = get_dataset(config, 'train', dgl_check=dgl_check, dataloader_args={'shuffle':True})
    val_dataset = get_dataset(config, 'val', dgl_check=dgl_check)
    return train_dataset, val_dataset

def get_test_dataset(config:dict, dgl_check=True):
    test_dataset = get_dataset(config, 'test', dgl_check=dgl_check)
    return test_dataset
    




