""" Active learning of ligands"""
from argparse import ArgumentParser
import pickle
from os import mkdir
from os.path import isdir, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

import active_learning_models
import data_processing
import tensor_ops
import plotting_functions

device = torch.device("cpu")
dtype = torch.float64

def convert_tensor_id_to_ligand_name(tensor_id):
    ligand_indexing_minimum = 1
    return tensor_id + ligand_indexing_minimum


def save_to_excel(ligand_name, 
                  response_name,
                  mean_values, 
                  lower_limits, 
                  upper_limits, 
                  filepath):
    data_frame = pd.DataFrame({
        'Ligand Index': ligand_name,
        f'Predicted {response_name}': mean_values,
        'Lower confidence limit': lower_limits,
        'Upper confidence limit': upper_limits, 
    })
    print('Saving to ', filepath)
    data_frame.to_excel(filepath)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', help='config yaml')
    parser.add_argument('-it', '--interactive', 
                        help='Set if required to run in interactive mode')
    args = parser.parse_args()
    interactive_run = args.interactive

    with open(args.config, "r") as fp:
        configs = yaml.load(fp, Loader=yaml.FullLoader)
    X_train_path = configs['train_data']['X']
    y_train_path = configs['train_data']['y']
    X_test_path = configs['test_data']['X']
    acq_fn_label = configs['acquisition_function']['label']
    acq_fn_hyperparams =  configs['acquisition_function'].get('hyperparameters')
    n_candidates = configs['number_of_candidates']
    output_base_dir = configs['output_dir']
    ylabel = configs['response']
    random_seed = configs['random_seed']
    plot_color = configs['plotting_color'] 
    visualization_dim = configs.get('visualization_dim', 0)
    
    X_train = torch.from_numpy(pickle.load(open(X_train_path, "rb"))).type(dtype)
    y_train = torch.from_numpy(pickle.load(open(y_train_path, "rb"))
                        ).type(dtype).reshape(-1, 1)
    X_test = torch.from_numpy(pickle.load(open(X_test_path, "rb"))).type(dtype)
    X_train = tensor_ops.normalize_tensor(X_train, dim=0)
    y_train = tensor_ops.normalize_tensor(y_train, dim=0)
    X_test = tensor_ops.normalize_tensor(X_test,
                                         mean_=X_train['mean'],
                                         std_deviation_=X_train['std'])
    
    gpr_model, gpr_mll = active_learning_models.train_gpr_model(
                                                        X=X_train['normalized'], 
                                                        y=y_train['normalized'])
    acq_vals = active_learning_models.get_new_points_acq_func_vals(
                                    model=gpr_model, 
                                    acq_fn_label=acq_fn_label,
                                    acq_fn_hyperparams=acq_fn_hyperparams, 
                                    new_points=X_test['normalized'],
                                    best_response=y_train['normalized'].max())
    top_acq_vals = torch.topk(acq_vals, n_candidates)
    print(f'Top {n_candidates} candidates for {ylabel}')
    
    top_ligand_names, top_ligands_acq_vals = [], []
    for _ in range(n_candidates):
        ligand_name = convert_tensor_id_to_ligand_name(top_acq_vals.indices[_])
        ligand_name = int(ligand_name)
        acq_val = top_acq_vals.values[_].detach().numpy()
        print(f'Ligand {ligand_name}: acquisiton value {acq_val}')
        top_ligand_names.append(ligand_name)
        top_ligands_acq_vals.append(acq_val)
    
    plt.rcParams['svg.fonttype'] = 'none'
    plt.bar(range(len(top_ligand_names)), top_ligands_acq_vals, color=plot_color)
    plt.xlabel('Ligand Index', fontsize=20)
    plt.ylabel('Acquisition Value', fontsize=20)
    plt.xticks(range(len(top_ligand_names)), labels=top_ligand_names, fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()

    posterior_means = active_learning_models.get_posterior_mean(
                                                        model=gpr_model,
                                                        X=X_test['normalized'])
    posterior_means = tensor_ops.scaleup_tensor(
                                   posterior_means, 
                                   mean_=y_train['mean'],
                                   std_deviation_=y_train['std'])
    lower, upper = active_learning_models.get_lower_upper_confidence(
                                                        model=gpr_model,
                                                        X=X_test['normalized'])

    lower = tensor_ops.scaleup_tensor(
                                   lower, 
                                   mean_=y_train['mean'],
                                   std_deviation_=y_train['std'])
    upper = tensor_ops.scaleup_tensor(
                                   upper, 
                                   mean_=y_train['mean'],
                                   std_deviation_=y_train['std'])
    
    plotting_functions.plot_confidence_region(means=posterior_means,
                                              lowers=lower, 
                                              uppers=upper,
                                              x_label='Virtual screen ligands',
                                              y_label=f'Predicted {ylabel}',
                                              legend='95% Confidence Region')

    save_to_excel(ligand_name=[convert_tensor_id_to_ligand_name(id) 
                               for id, _ in enumerate(posterior_means)],
                  mean_values=posterior_means.numpy()[:, 0],
                  lower_limits=lower,
                  upper_limits=upper,
                  response_name=ylabel,
                  filepath=join(output_base_dir, f'screen_predicted_{ylabel}.xlsx'))


    
    

