from metrics.hhc import hhc_edgefeat_compute_accuracy
from metrics.mcp import mcp_edgefeat_compute_accuracy
from metrics.sbm import sbm_edgefeat_compute_accuracy
from metrics.tsp import tsp_edgefeat_compute_f1
from models.base_model import GNN_Abstract_Base_Class 

def get_fgnn_edge_metric(problem):
    if problem=='tsp':
        return tsp_edgefeat_compute_f1
    elif problem in ('tsp_mt', 'tsp_bgnn'):
        return tsp_edgefeat_compute_f1
    elif problem=='mcp':
        return mcp_edgefeat_compute_accuracy
    elif problem=='sbm':
        return sbm_edgefeat_compute_accuracy
    elif problem=='hhc':
        return hhc_edgefeat_compute_accuracy
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
        return mcp_edgefeat_compute_accuracy
    elif problem=='hhc':
        raise hhc_edgefeat_compute_accuracy
    elif problem=='sbm':
        raise sbm_edgefeat_compute_accuracy
    else:
        raise NotImplementedError(f"Metric for dgl edge problem {problem} has not been implemented.")

def get_dgl_node_metric(problem):
    raise NotImplementedError()

def setup_metric(pl_model: GNN_Abstract_Base_Class, config: dict, soft=True)-> None:
    problem = config['problem']
    use_dgl = config['arch']['use_dgl']
    embed = config['arch']['embedding']
    eval = config['arch']['eval']
    try:
        if use_dgl and eval=='edge':
            metric_fn = get_edgefeat_metric(problem)
        elif use_dgl and eval=='node':
            metric_fn = get_dgl_node_metric(problem)
        elif not(use_dgl) and eval=='edge':
            metric_fn = get_fgnn_edge_metric(problem)
        elif not(use_dgl) and eval=='node':
            metric_fn = get_fgnn_node_metric(problem)
        pl_model.attach_metric_function(metric_fn, start_using_metric=True)
    except NotImplementedError as ne:
        if not soft:
            raise ne
        print(f"The metric for {problem=} with {use_dgl=} and {embed=} has not been implemented. I'll let it go anyways, but additional metrics won't be saved.")



