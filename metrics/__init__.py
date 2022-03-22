from metrics.preprocess import edgefeat_converter, fulledge_converter
from metrics.common import fulledge_compute_f1, edgefeat_compute_f1
from metrics.tsp import tsp_edgefeat_compute_f1
from models.base_model import GNN_Abstract_Base_Class 

def get_fulledge_metric(problem):
    if problem=='tsp':
        return tsp_edgefeat_compute_f1
    elif problem in ('tsp_mt', 'tsp_bgnn'):
        return tsp_edgefeat_compute_f1
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
        return tsp_edgefeat_compute_f1
    elif problem=='tsp_bgnn':
        return tsp_edgefeat_compute_f1
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

def get_preprocessing(embed, eval):
    if embed=='edge':
        if eval=='edge':
            return edgefeat_converter
        elif eval=='fulledge':
            return fulledge_converter
        else:
            raise NotImplementedError(f"Unknown eval '{eval}' for embedding type 'edge'.")
    else:
        raise NotImplementedError(f"Embed {embed} not implemented.")

def assemble_metric_function(preprocess_function, eval_function):
    def final_function(raw_scores, target, **kwargs):
        inferred, targets = preprocess_function(raw_scores, target, **kwargs)
        result = eval_function(inferred, targets)
        return result
    return final_function

def setup_metric(pl_model: GNN_Abstract_Base_Class, config: dict, soft=True)-> None:
    problem = config['problem']
    embed = config['arch']['embedding']
    eval = config['arch']['eval']
    try:
        preprocess_function = get_preprocessing(embed, eval)

        if eval=='edge':
            eval_fn = get_edgefeat_metric(problem)
        elif eval=='fulledge':
            eval_fn = get_fulledge_metric(problem)
        elif eval=='node':
            eval_fn = get_node_metric(problem)
        
        metric_fn = assemble_metric_function(preprocess_function=preprocess_function, eval_function=eval_fn)
        pl_model.attach_metric_function(metric_fn, start_using_metric=True)
    except NotImplementedError as ne:
        if not soft:
            raise ne
        print(f"There was a problem with the setup metric. I'll let it go anyways, but additional metrics won't be saved. Error is stated below:\n {ne}")



