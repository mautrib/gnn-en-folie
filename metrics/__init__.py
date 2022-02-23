from metrics.tsp import compute_f1
from models.base_model import GNN_Abstract_Base_Class 

def get_metric(problem):
    if problem=='tsp':
        return compute_f1
    else:
        raise NotImplementedError(f"Metric for problem {problem} has not been implemented.")

def setup_metric(pl_model: GNN_Abstract_Base_Class, config: dict)-> None:
    problem = config['problem']
    metric_fn = get_metric(problem)
    pl_model.attach_metric_function(metric_fn, start_using_metric=True)


