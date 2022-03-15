from metrics.hhc import hhc_dgl_edge_compute_accuracy_unbatch, hhc_fgnn_edge_compute_accuracy
from metrics.mcp import mcp_dgl_edge_compute_accuracy_unbatch, mcp_fgnn_edge_compute_accuracy
from metrics.sbm import sbm_dgl_edge_compute_accuracy_unbatch, sbm_fgnn_edge_compute_accuracy
from metrics.tsp import tsp_edgefeat_compute_f1, tsp_fgnn_edge_compute_f1, tsp_dgl_edge_compute_f1, tsp_mt_edge_compute_f1
from models.base_model import GNN_Abstract_Base_Class 

def get_fgnn_edge_metric(problem):
    if problem=='tsp':
        return tsp_fgnn_edge_compute_f1
    elif problem in ('tsp_mt', 'tsp_bgnn'):
        return tsp_mt_edge_compute_f1
    elif problem=='mcp':
        return mcp_fgnn_edge_compute_accuracy
    elif problem=='sbm':
        return sbm_fgnn_edge_compute_accuracy
    elif problem=='hhc':
        return hhc_fgnn_edge_compute_accuracy
    else:
        raise NotImplementedError(f"Metric for fgnn edge problem {problem} has not been implemented.")

def get_fgnn_node_metric(problem):
    raise NotImplementedError()

def get_dgl_edge_metric(problem):
    if problem=='tsp':
        return tsp_edgefeat_compute_f1
    elif problem=='mcp':
        return mcp_dgl_edge_compute_accuracy_unbatch
    elif problem=='hhc':
        return hhc_dgl_edge_compute_accuracy_unbatch
    elif problem=='sbm':
        return sbm_dgl_edge_compute_accuracy_unbatch
    else:
        raise NotImplementedError(f"Metric for dgl edge problem {problem} has not been implemented.")

def get_dgl_node_metric(problem):
    raise NotImplementedError()

def setup_metric(pl_model: GNN_Abstract_Base_Class, config: dict, soft=True)-> None:
    problem = config['problem']
    use_dgl = config['arch']['use_dgl']
    embed = config['arch']['embedding']
    try:
        if use_dgl and embed=='edge':
            metric_fn = get_dgl_edge_metric(problem)
        elif use_dgl and embed=='node':
            metric_fn = get_dgl_node_metric(problem)
        elif not(use_dgl) and embed=='edge':
            metric_fn = get_fgnn_edge_metric(problem)
        elif not(use_dgl) and embed=='node':
            metric_fn = get_fgnn_node_metric(problem)
        pl_model.attach_metric_function(metric_fn, start_using_metric=True)
    except NotImplementedError as ne:
        if not soft:
            raise ne
        print(f"The metric for {problem=} with {use_dgl=} and {embed=} has not been implemented. I'll let it go anyways, but additional metrics won't be saved.")



