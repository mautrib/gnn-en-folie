import sys, os
sys.path.append(os.getcwd())
import time
from copy import deepcopy
from commander import get_config
import numpy as np
import yaml
import wandb
import random

from string import ascii_letters,digits
NAME_CHARS = ascii_letters+digits

def get_config_specific(value, config=None):
    if config is None: config = BASE_CONFIG
    config = deepcopy(config)
    if PROBLEM == 'sbm':
        p_inter = C-value/2
        p_outer = C+value/2
        config['data']['test']['problems'][PROBLEM]['p_inter'] = p_inter
        config['data']['test']['problems'][PROBLEM]['p_outer'] = p_outer
    elif PROBLEM in ('mcp', 'hhc'):
        config['data']['test']['problems'][PROBLEM][VALUE_NAME] = value
    else:
        raise NotImplementedError(f'Problem {PROBLEM} config modification not implemented.')
    return config

def get_values(trainer):
    logged = trainer.logged_metrics
    loss_name = 'test_loss/dataloader_idx_{}'
    metrics_name = 'test.metrics/dataloader_idx_{}'
    total_dict = {}
    for i, value in enumerate(VALUES):
        loss_value = logged[loss_name.format(i)]
        metrics_value = logged[metrics_name.format(i)]
        values_dict = {
            'loss': loss_value,
            'metrics': metrics_value
        }
        total_dict[f'{value:.4f}'] = values_dict
    return total_dict

def get_train_value(run):
    config = run.config
    if PROBLEM == 'sbm':
        p_outer = config['data']['train']['problems'][PROBLEM]['p_outer']
        p_inter = config['data']['train']['problems'][PROBLEM]['p_inter']
        value = p_outer-p_inter
    elif PROBLEM in ('mcp', 'hhc'):
        value = config['data']['train']['problems'][PROBLEM][VALUE_NAME]
    else:
        raise NotImplementedError(f'Problem {PROBLEM} config modification not implemented.')
    return value

class ExpLauncher():
    BASE_STR = 'sbatch {}/slurm_expe.batch {}'
    TIMEOUT = 60 * 60 #60mins to override the expe

    def __init__(self, base_run, config, api_object, threadID, path, name = ''):
        super().__init__()
        self.base_run = base_run
        self.train_value = get_train_value(base_run)
        self.project = config['project'] + f"_{config['problem']}"
        self.api = api_object
        self.threadID = threadID

        if name=='':
            name = ''.join(random.choice(NAME_CHARS) for _ in range(10))
        self.name = name
        self.path = path
        self.config_filename = '_temp-' + name + '.yaml'
        self.config_fullpath = os.path.join(self.path, self.config_filename)

        self.prepare_config(config)

        self.already_executed = False
        self.execute_time = 0
    
    def prepare_config(self, config):
        config = deepcopy(config)
        config['train_value'] = self.train_value
        config['wandb_source_project'] = self.base_run.project
        config['wandb_source_id'] = self.base_run.id
        config['creator'] = self.name
        self.config = config

    def export_config(self):
        with open(self.config_fullpath, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def remove_files(self):
        try:
            os.remove(self.config_fullpath)
        except FileNotFoundError:
            pass

    def run_command(self):
        cur_str = self.BASE_STR.format(self.path, self.config_filename)
        print(f"Thread {self.name} launching command '{cur_str}'", flush=True)
        os.system(cur_str)
        self.already_executed = True
        self.execute_time = time.time()
    
    def override(self):
        return time.time() - self.execute_time > self.TIMEOUT
    
    def launch_expe(self):
        self.export_config()
        self.run_command()

    def deserves_living(self)->bool:
        """Returns True if need to do this run"""
        try:
            runs = self.api.runs(self.project)
            for run in runs:
                if run.config['train_value']==self.train_value:
                    if run.state=='failed':
                        run.delete()
                    elif run.state=='finished':
                        return False
                    else:
                        if run.config['creator']!=self.name: #In case another thread works on this project
                            return False
        except Exception:
            pass
        return True

    def check_and_execute_expe(self) -> bool:
        """Returns True if the run was finished successfully. If it still needs to compute, returns False. If the run found with this train_value has failed, it relaunches it."""
        try:
            runs = self.api.runs(self.project)
            found = False
            for run in runs:
                if run.config['train_value']==self.train_value:
                    found = True
                    if run.state=='failed':
                        run.delete()
                        self.already_executed = False
                    elif run.state=='finished':
                        return True
                    elif run.state=='running':
                        pass
        except Exception:
            found = True
        if found:
            if not self.already_executed: self.launch_expe()
        else:
            if self.override():
                self.launch_expe()
        return False

if __name__=='__main__':

    #VALUES_DEPENDING ON ABOVE
    BASE_PATH = 'scripts/'
    CONFIG_FILE_NAME = f'slurm_grid_config.yaml'
    CONFIG_FILE = os.path.join(BASE_PATH, CONFIG_FILE_NAME)
    BASE_CONFIG = get_config(CONFIG_FILE)

    #CONFIG DEPENDING
    PROBLEM = BASE_CONFIG['problem']
    WANDB_MODELS_PROJECT = BASE_CONFIG['wandb_source_project'] + f"_{PROBLEM}"
    NUM_TASKS = BASE_CONFIG['num_tasks']
    SLEEP_TIME = BASE_CONFIG['sleep_time']

    print(f"Working on problem '{PROBLEM}'")
    if PROBLEM == 'mcp':
        VALUE_NAME = 'clique_size'
        VALUES = range(5,20)
    elif PROBLEM == 'sbm':
        VALUE_NAME = 'dc'
        VALUES = np.linspace(0,6,25)
        C=3
    elif PROBLEM == 'hhc':
        VALUE_NAME = 'fill_param'
        l_musquare = np.linspace(0,25,26)
        VALUES = np.sqrt(l_musquare)
    else:
        raise NotImplementedError(f"Problem {PROBLEM} not implemented.")

    l_threads = []

    wapi = wandb.Api()
    runs = wapi.runs(WANDB_MODELS_PROJECT)
    runids = [run.id for run in runs]
    todo = [run.id for run in runs]
    done = []
    total_runs = len(runs)

    cur_thread_num = min(len(todo), NUM_TASKS)

    global_thread_num = 0
    while len(done)!=len(runids):
        
        print(f"TODO : {len(todo)}")
        #SETUP THREADS
        while (len(l_threads) < cur_thread_num) and len(todo)!=0:
            run_id = todo.pop()
            run = wapi.run(os.path.join(WANDB_MODELS_PROJECT, run_id))
            thread = ExpLauncher(run, BASE_CONFIG, wapi, global_thread_num, BASE_PATH)
            l_threads.append(thread)
            if not thread.deserves_living():
                l_threads.pop()
                done.append(run_id)
        
        #EXECUTE THREADS
        to_remove = []
        for i, thread in enumerate(l_threads):
            finished = thread.check_and_execute_expe()
            if finished:
                to_remove.append(i)
            time.sleep(10) #For the wandb api problems
        
        
        #REMOVE FINISHED ONES
        l_finished_threads = [thread for i, thread in enumerate(l_threads) if (i in to_remove)]
        for thread in l_finished_threads:
            thread.remove_files()
            print(f"Finished thread {thread.name}, train value {thread.train_value}")
            done.append(thread.base_run.id)

        l_threads = [thread for i, thread in enumerate(l_threads) if (not i in to_remove)]

        #Wait
        time.sleep(SLEEP_TIME * 60)


            

        
            

        
        

                


    



