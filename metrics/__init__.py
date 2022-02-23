from metrics.mcp import mcp_compute_accuracy
from metrics.tsp import tsp_compute_f1
from models.base_model import GNN_Abstract_Base_Class 

def get_fgnn_edge_metric(problem):
    if problem=='tsp':
        return tsp_compute_f1
    elif problem=='mcp':
        return mcp_compute_accuracy
    else:
        raise NotImplementedError(f"Metric for problem {problem} has not been implemented.")

def get_fgnn_node_metric(problem):
    raise NotImplementedError()

def get_dgl_edge_metric(problem):
    raise NotImplementedError()

def get_dgl_node_metric(problem):
    raise NotImplementedError()

def setup_metric(pl_model: GNN_Abstract_Base_Class, config: dict)-> None:
    problem = config['problem']
    use_dgl = config['arch']['use_dgl']
    embed = config['arch']['embedding']
    if use_dgl and embed=='edge':
        metric_fn = get_dgl_edge_metric(problem)
    elif use_dgl and embed=='node':
        metric_fn = get_dgl_node_metric(problem)
    elif not(use_dgl) and embed=='edge':
        metric_fn = get_fgnn_edge_metric(problem)
    elif not(use_dgl) and embed=='node':
        metric_fn = get_fgnn_node_metric(problem)
    pl_model.attach_metric_function(metric_fn, start_using_metric=True)


