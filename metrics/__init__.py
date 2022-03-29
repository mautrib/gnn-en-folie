from metrics.preprocess import edgefeat_converter, fulledge_converter
from metrics.common import fulledge_compute_f1, edgefeat_compute_f1
from metrics.tsp import tsp_edgefeat_converter_sparsify, tsp_fulledge_compute_f1
from models.base_model import GNN_Abstract_Base_Class 

def get_fulledge_metric(problem):
    if problem=='tsp':
        return tsp_fulledge_compute_f1
    elif problem=='mcp':
        return fulledge_compute_f1
    elif problem=='sbm':
        return fulledge_compute_f1
    elif problem=='hhc':
        return fulledge_compute_f1
    else:
        raise NotImplementedError(f"Metric for fgnn edge problem {problem} has not been implemented.")

def get_edgefeat_metric(problem):
    if problem=='tsp':
        return edgefeat_compute_f1
    elif problem=='mcp':
        return edgefeat_compute_f1
    elif problem=='hhc':
        raise edgefeat_compute_f1
    elif problem=='sbm':
        raise edgefeat_compute_f1
    else:
        raise NotImplementedError(f"Metric for dgl edge problem {problem} has not been implemented.")

def get_node_metric(problem):
    raise NotImplementedError()

def get_metric(eval, problem):
    if eval=='edge':
        eval_fn = get_edgefeat_metric(problem)
    elif eval=='fulledge':
        eval_fn = get_fulledge_metric(problem)
    elif eval=='node':
        eval_fn = get_node_metric(problem)
    else:
        raise NotImplementedError(f"Eval method {eval} not implemented")
    return eval_fn

def get_preprocessing(embed, eval, problem):
    if embed=='edge':
        if eval=='edge':
            if problem=='tsp':
                return tsp_edgefeat_converter_sparsify
            elif problem in ('mcp','sbm'):
                return edgefeat_converter
            else:
                raise NotImplementedError(f"Preprocessing for {embed=}, {eval=}, {problem=} not implemented")
        elif eval=='fulledge':
            if problem in ('mcp','sbm','tsp'):
                return fulledge_converter
            else:
                raise NotImplementedError(f"Preprocessing for {embed=}, {eval=}, {problem=} not implemented")
        else:
            raise NotImplementedError(f"Unknown eval '{eval}' for embedding type 'edge'.")
    else:
        raise NotImplementedError(f"Embed {embed} not implemented.")

def get_preprocess_additional_args(problem: str, config: dict):
    if problem=='tsp':
        return {'sparsify': config['data']['train']['sparsify']}
    return {}

def assemble_metric_function(preprocess_function, eval_function, preprocess_additional_args=None):
    if preprocess_additional_args is None:
        preprocess_additional_args = {}
    def final_function(raw_scores, target, **kwargs):
        inferred, targets = preprocess_function(raw_scores, target, **kwargs, **preprocess_additional_args)
        result = eval_function(inferred, targets)
        return result
    return final_function

def setup_metric(pl_model: GNN_Abstract_Base_Class, config: dict, soft=True)-> None:
    problem = config['problem']
    embed = config['arch']['embedding']
    eval = config['arch']['eval']
    try:
        preprocess_function = get_preprocessing(embed, eval, problem)
        eval_fn = get_metric(eval, problem)
        preprocess_additional_args = get_preprocess_additional_args(problem, config)
        metric_fn = assemble_metric_function(preprocess_function=preprocess_function, eval_function=eval_fn, preprocess_additional_args=preprocess_additional_args)
        pl_model.attach_metric_function(metric_fn, start_using_metric=True)
    except NotImplementedError as ne:
        if not soft:
            raise ne
        print(f"There was a problem with the setup metric. I'll let it go anyways, but additional metrics won't be saved. Error stated is: {ne}")



