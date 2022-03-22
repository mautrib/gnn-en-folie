from metrics.preprocess import edgefeat_converter, fulledge_converter
from metrics.common import edgefeat_compute_accuracy, fulledge_compute_accuracy
from metrics.tsp import tsp_edgefeat_compute_f1
from models.base_model import GNN_Abstract_Base_Class 

def get_fgnn_edge_metric(problem):
    if problem=='tsp':
        return tsp_edgefeat_compute_f1
    elif problem in ('tsp_mt', 'tsp_bgnn'):
        return tsp_edgefeat_compute_f1
    elif problem=='mcp':
        return fulledge_compute_accuracy
    elif problem=='sbm':
        return fulledge_compute_accuracy
    elif problem=='hhc':
        return fulledge_compute_accuracy
    else:
        raise NotImplementedError(f"Metric for fgnn edge problem {problem} has not been implemented.")

def get_fgnn_node_metric(problem):
    raise NotImplementedError()

def get_edgefeat_metric(problem):
    if problem=='tsp':
        return tsp_edgefeat_compute_f1
    elif problem=='tsp_bgnn':
        return tsp_edgefeat_compute_f1
    elif problem=='mcp':
        return edgefeat_compute_accuracy
    elif problem=='hhc':
        raise edgefeat_compute_accuracy
    elif problem=='sbm':
        raise edgefeat_compute_accuracy
    else:
        raise NotImplementedError(f"Metric for dgl edge problem {problem} has not been implemented.")

def get_dgl_node_metric(problem):
    raise NotImplementedError()

def get_preprocessing(eval):
    if eval=='edge':
        return edgefeat_converter
    elif eval=='fulledge':
        return fulledge_converter
    else:
        raise NotImplementedError(f"H")

def assemble_metric_function(preprocess_function, eval_function):
    def final_function(raw_scores, target, **kwargs):
        inferred, targets = preprocess_function(raw_scores, target, **kwargs)
        result = eval_function(inferred, targets)
        return result
    return final_function

def setup_metric(pl_model: GNN_Abstract_Base_Class, config: dict, soft=True)-> None:
    problem = config['problem']
    use_dgl = config['arch']['use_dgl']
    embed = config['arch']['embedding']
    eval = config['arch']['eval']
    try:
        preprocess_function = get_preprocessing(eval)

        if use_dgl and embed=='edge':
            eval_fn = get_edgefeat_metric(problem)
        elif use_dgl and embed=='node':
            eval_fn = get_dgl_node_metric(problem)
        elif not(use_dgl) and embed=='edge':
            eval_fn = get_fgnn_edge_metric(problem)
        elif not(use_dgl) and embed=='node':
            eval_fn = get_fgnn_node_metric(problem)
        
        metric_fn = assemble_metric_function(preprocess_function=preprocess_function, eval_function=eval_fn)
        pl_model.attach_metric_function(metric_fn, start_using_metric=True)
    except NotImplementedError as ne:
        if not soft:
            raise ne
        print(f"There was a problem with the setup metric. I'll let it go anyways, but additional metrics won't be saved. Error is stated below:\n {ne}")



