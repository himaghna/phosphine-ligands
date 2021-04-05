""" Active learning of ligands"""
from argparse import ArgumentParser
import pickle
from os import mkdir
from os.path import isdir, join

import numpy as np
import torch
import yaml

import active_learning_models
import data_processing
import plotting_functions
import tensor_ops


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
dtype = torch.float64


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='config yaml')
    parser.add_argument('-it', '--interactive', 
                        help='Set if required to run in interactive mode')
    args = parser.parse_args()
    interactive_run = args.interactive
    with open(args.config, "r") as fp:
        configs = yaml.load(fp, Loader=yaml.FullLoader)
    X_path = configs['X']
    y_path = configs['y']
    acq_fn_label = configs['acquisition_function']
    descriptor_names_path = configs['descriptor_names']
    output_base_dir = configs['output_dir']
    ylabel = configs['response']
    proportion_of_dataset_for_seeding = configs['proportion_of_dataset_for_seeding']                         
    random_seed = configs['random_seed']
    visualization_dim = configs.get('visualization_dim', 0)

    X = torch.from_numpy(pickle.load(open(X_path, "rb"))).type(dtype)
    y = torch.from_numpy(pickle.load(open(y_path, "rb"))
                        ).type(dtype).reshape(-1, 1)
    y_range = float(torch.max(y).numpy() - torch.min(y).numpy())
    X = tensor_ops.normalize_tensor(X, dim=0)
    y = tensor_ops.normalize_tensor(y, dim=0)
    opt_bounds = torch.stack([X['normalized'].min(dim=0).values, 
                              X['normalized'].max(dim=0).values])
    descriptor_names = pickle.load(open(descriptor_names_path, "rb"))
    xlabel = descriptor_names[visualization_dim]
    X_train, X_test, y_train, y_test = tensor_ops.train_test_split(
                            X=X['normalized'],
                            y=y['normalized'], 
                            train_fraction=proportion_of_dataset_for_seeding, 
                            random_seed=random_seed)
    print(f'Using {X_train.shape[0]} / {X["normalized"].shape[0]}' 
           ' points for initial seeding')
    
    maes = []
    y_test_means = [float(torch.mean(y_test).numpy())]
    gpr_model, gpr_mll = active_learning_models.train_gpr_model(X=X_train, 
                                                                y=y_train)
    mean_absolute_error = tensor_ops.get_mean_absolute_error(X_test=X_test, 
                                                             y_test=y_test, 
                                                             gpr_model=gpr_model, 
                                                             y_mean=y['mean'], 
                                                             y_scale=y['std'])
    maes.append(mean_absolute_error)
    plotting_functions.plot_testing(gpr_model, 
                                    X_test=X_test, 
                                    X_train=X_train, 
                                    y_train=y_train,
                                    y_test=y_test,
                                    xlabel=xlabel,
                                    ylabel=ylabel,
                                    visualization_dim=visualization_dim,
                                    interactive_run=interactive_run,
                                    mean_absolute_error=mean_absolute_error,
                                    out_dir=output_base_dir, 
                                    out_fname='initial')
    
    for _ in range(configs.get('n_optimization_steps')):
        best_response = y_train.max()
        acq_vals = active_learning_models.get_new_points_acq_func_vals(
                                                    model=gpr_model, 
                                                    acq_fn_label=acq_fn_label, 
                                                    new_points=X_test,
                                                    best_response=best_response)
        (X_new, y_new), new_point_id  = tensor_ops.get_new_point_to_acquire(
                                                    test_data=(X_test, y_test),
                                                    acq_vals=acq_vals)
        (X_train, y_train), (X_test, y_test) = tensor_ops.move_test_to_train(
                                               training_data=(X_train, y_train),
                                               test_data=(X_test, y_test),  
                                               id_of_point_to_move=new_point_id)  
        gpr_model, gpr_mll = active_learning_models.train_gpr_model(X_new,
                                                                    y_new, 
                                                                    model=gpr_model)

        mean_absolute_error = tensor_ops.get_mean_absolute_error(
                                                            X_test=X_test, 
                                                            y_test=y_test, 
                                                            gpr_model=gpr_model, 
                                                            y_mean=y['mean'], 
                                                            y_scale=y['std'])
        maes.append(mean_absolute_error)
        y_test_means.append(float(torch.mean(y_test).numpy()))
        plotting_functions.plot_testing(gpr_model, 
                                        X_test=X_test, X_train=X_train, 
                                        y_test=y_test, y_train=y_train,
                                        xlabel=xlabel, ylabel=ylabel,
                                        X_new=X_new, y_new=y_new, 
                                        mean_absolute_error=mean_absolute_error,
                                        visualization_dim=visualization_dim,
                                        out_dir=output_base_dir, 
                                        out_fname='image'+str(_+1))
    plotting_functions.plot_maes(maes, 
                                 out_dir=output_base_dir, 
                                 mae_scale=y_range, 
                                 mae_scale_label='Range',
                                 response_name=configs.get('response').upper(),
                                 interactive_run=interactive_run)
    plotting_functions.plot_y_test_means(
                                y_test_means,
                                out_dir=output_base_dir, 
                                response_name=configs.get('response').upper(),
                                interactive_run=interactive_run)
        
    # plt.plot([_ for _ in range(configs.get('n_optimization_steps'))], max_val, 
    #          'go--', linewidth=2, markersize=12)
    # plt.fill_between([_ for _ in range(configs.get('n_optimization_steps'))], 
    #                  lower_confidence, upper_confidence, 
    #                  alpha=0.5, label = '95% Credibility')
    # plt.show()

if __name__ == "__main__":
    main()