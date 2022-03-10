import os
import argparse
import numpy as np
import pandas as pd
import wandb
wapi = wandb.Api()
import tqdm

PROJECT_NAME = 'custom_charts'
RUN_NAME = 'grid_chart_{PROBLEM}'


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Creates a chart from a W&B grid project repository.')
    parser.add_argument('problem', metavar='problem', choices = ('mcp','hhc','sbm'), help='Need to choose an experiment')
    args = parser.parse_args()

    PROBLEM = args.problem
    RUN_NAME = RUN_NAME.format(PROBLEM=PROBLEM)
    
    #WANDB
    WANDB_ENTITY = 'mautrib'
    WANDB_REPO_PROJECT = f"grid_{PROBLEM}"

    grid_runs = wapi.runs(os.path.join(WANDB_ENTITY, WANDB_REPO_PROJECT))
    total_runs = len(grid_runs)

    df_dict = {
        'train_value': [],
        'test_value': [],
        'accuracy': [],
        'test_loss': []
    }

    for grid_run in tqdm.tqdm(grid_runs, total=total_runs):
        summary = grid_run.summary
        train_value = summary['train_value']
        for key, metrics in summary['values_logged'].items():
            test_value = float(key)
            accuracy = float(metrics['metrics']['accuracy'])
            test_loss = float(metrics['loss'])
            df_dict['train_value'].append(train_value)
            df_dict['test_value'].append(test_value)
            df_dict['accuracy'].append(accuracy)
            df_dict['test_loss'].append(test_loss)
    df = pd.DataFrame.from_dict(df_dict)

    
    train_values = np.unique(df['train_value'])
    test_values = np.unique(df['test_value'])
    n_train = len(train_values)
    n_test = len(test_values)
    #assert n_train==n_test #If data should be square
    #print(f'Train, test : {n_train}, {n_test}')
    train_sorted = sorted(train_values)
    test_sorted = sorted(test_values)
    #print('Sorted train : ', train_sorted)
    #print('Sorted test: ', test_sorted)

    hm_loss = np.zeros((n_train,n_test), dtype=float)
    hm_accuracy = np.zeros((n_train,n_test), dtype=float)

    for i,train_value in enumerate(train_sorted):
        for j,test_value in enumerate(test_sorted):
            ij_df = df[ (df['train_value']==train_value) & (df['test_value']==test_value) ]
            hm_loss[i,j] = ij_df['test_loss'].mean()
            hm_accuracy[i,j] = ij_df['accuracy'].mean()

    for _,line in tqdm.tqdm(df.iterrows(), total = len(df)):
        train_value = line['train_value']
        test_value = line['test_value']
        train_index = train_sorted.index(train_value)
        test_index = test_sorted.index(test_value)
        hm_loss[train_index, test_index] = line['test_loss']
        hm_accuracy[train_index, test_index] = line['accuracy']

    run = wandb.init(project=PROJECT_NAME, name=RUN_NAME)
    wandb.log({'Accuracy heatmap' : wandb.plots.HeatMap(x_labels=test_sorted, y_labels=train_sorted, matrix_values=hm_accuracy, show_text=True)})
    wandb.log({'Losses heatmap' : wandb.plots.HeatMap(x_labels=test_sorted, y_labels=train_sorted, matrix_values=hm_loss, show_text=True)})

    wandb.finish()