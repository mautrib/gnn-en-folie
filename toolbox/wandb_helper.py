import os
import yaml
import wandb

def find_exp_dir(base_path, exp_name):
    dirs = [name for name in os.listdir(base_path) if exp_name in name]
    if len(dirs)==0:
        raise RuntimeError(f"Experiment {exp_name} not found in directory {base_path}.")
    elif len(dirs)>1:
        raise RuntimeError(f"Found multiple directories corresponding to experiment {exp_name} : {dirs}.")
    return dirs[0]

def get_config_from_dir(directory):
    filename = os.path.join(directory, "files/config.yaml")
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_config(entity, project, run_id):
    exp_path = os.path.join(entity, project, run_id)
    wapi = wandb.Api()
    run = wapi.run(exp_path)
    return run.config

def download_model(project, run_id, entity='', version='best'):
    """
    Downloads a model from a wandb run
    """
    exp_path = os.path.join(entity, project, run_id)
    wapi = wandb.Api()
    run = wapi.run(exp_path)
    model_name = f'model-{run_id}:{version}'
    model_artifact_name = os.path.join(entity, project, model_name)
    print(f"Getting model artifact from : {model_artifact_name}")
    artifact = wapi.artifact(model_artifact_name)
    art_dir = artifact.download('_temp')
    model_dir = os.path.join(art_dir, 'model.ckpt')
    config = run.config
    return config, model_dir

# def get_model(project, run_id, entity='', version='best'):
#     """
#     Retrieves the best model from a wandb run
#     """
#     exp_path = os.path.join(entity, project, run_id)
#     wapi = wandb.Api()
#     run = wapi.run(exp_path)
#     model_name = f'model-{run_id}:{version}'
#     model_artifact_name = os.path.join(entity, project, model_name)
#     print(f"Getting model artifact from : {model_artifact_name}")
#     artifact = wapi.artifact(model_artifact_name)
#     art_dir = artifact.download('_temp')
#     model_dir = os.path.join(art_dir, 'model.ckpt')
#     config = run.config
#     pl_model = load_model(config, model_dir)
#     shutil.rmtree(art_dir)
#     return pl_model

def find_entity():
    """
    Tries to find the default entity (wandb doesn't make it easy)
    """
    project = '__temp'
    with wandb.init(reinit=True, project=project) as run:
        rid = run.id
        entity = run.entity
    wapi = wandb.Api()
    run = wapi.run(f"{entity}/{project}/{rid}")
    run.delete()
    return entity
