""" Active learning of ligands"""
from argparse import ArgumentParser
import pickle
from os import mkdir
from os.path import isdir, join

import numpy as np
import torch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
import yaml

import data_processing


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
dtype = torch.float64
VISUALIZATION_DIM = 1
INTERACTIVE = False

def get_gpr_model(X, y, model=None):
    """
    Fit a gpr model to the data or update the model to new data
    Params ::
    X: (sx1) Tensor: Covariates
    y: (sx1) Tensor: Observations
    model: PyTorch SingleTaskGP model: If model is passed, X and y are used to 
        update it. If None then model is trained on X and y. Default is None
    Return ::
    model: PyTorch SingleTaskGP model: Trained or updated model. 
        Returned in train mode
    mll: PyTorch MarginalLogLikelihood object: Returned in train mode
    """
    
    if model is None:
        # set up model
        model = SingleTaskGP(X, y)
    else:
        # update model with new observations
        print('Optimizing!')
        model = model.condition_on_observations(X, y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(X)
    # begin training
    model.train();
    mll.train();
    fit_gpytorch_model(mll);
    return model, mll

def optimize_loop(model, loss, X_train, y_train, X_test, y_test, bounds, xlabel):
    best_value = y_train.max()
    acq_func = ExpectedImprovement(model, best_f=best_value, maximize=True)
    acq_vals = acq_func(X_test.view((X_test.shape[0], 1, X_test.shape[1])))
    max_acqf_id = acq_vals.argmax()
    X_test_new, X_new = tensor_pop(inp_tensor=X_test,
        to_pop=max_acqf_id.cpu().numpy())
    y_test_new, y_new = tensor_pop(inp_tensor=y_test,
        to_pop=max_acqf_id.cpu().numpy())
    if INTERACTIVE:
        plot_acq_func(acq_func, 
                    X_test=X_test, X_train=X_train, xlabel=xlabel, X_new=X_new)
    gpr_model, gpr_mll = get_gpr_model(X_new, y_new, model=model)
    X_train_new = torch.cat((X_train, X_new))
    y_train_new = torch.cat((y_train, y_new))
    return {
        'model': gpr_model,
        'loss': gpr_mll,
        'X_train': X_train_new,
        'y_train': y_train_new,
        'X_test': X_test_new,
        'y_test': y_test_new,
        'X_new': X_new,
        'y_new': y_new,
    }



def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='config yaml')
    args = parser.parse_args()
    with open(args.config, "r") as fp:
        configs = yaml.load(fp, Loader=yaml.FullLoader)
    X_path = configs.get('X')
    y_path = configs.get('y')
    descriptor_names_path = configs.get('descriptor_names')
    output_base_dir = configs.get('output_dir')
    ylabel = configs.get('response')
    proportion_of_dataset_for_seeding = configs.get(
                                            'proportion_of_dataset_for_seeding')
    random_seed = configs.get('random_seed')

    X = torch.from_numpy(pickle.load(open(X_path, "rb"))).type(dtype)
    y = torch.from_numpy(pickle.load(open(y_path, "rb"))
                        ).type(dtype).reshape(-1, 1)
    descriptor_names = pickle.load(open(descriptor_names_path, "rb"))

    xlabel = descriptor_names[VISUALIZATION_DIM]
    y_range = float(torch.max(y).numpy() - torch.min(y).numpy())
    y_scale = y.std(dim=0)
    y_mean = y.mean(dim=0)
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0)
    X = (X - X_mean) / X_std
    y = (y - y_mean) / y_scale
    initial_data_size = int(proportion_of_dataset_for_seeding * X.shape[0])
    print(f'Using {initial_data_size} / {X.shape[0]} points for initial seeding')
    np.random.seed(random_seed)
    initial_idx = list(np.random.choice(X.shape[0], 
                       initial_data_size,
                       replace=False))
    X_test, X_train = tensor_pop(X, to_pop=initial_idx)
    y_test, y_train = tensor_pop(y, to_pop=initial_idx)
    maes = []
    y_test_means = [float(torch.mean(y_test).numpy())]
    gpr_model, gpr_mll = get_gpr_model(X=X_train, y=y_train)
    mean_absolute_error = get_mean_absolute_error(X_test=X_test, 
                                                  y_test=y_test, 
                                                  gpr_model=gpr_model, 
                                                  y_mean=y_mean, 
                                                  y_scale=y_scale)
    maes.append(mean_absolute_error)
    plot_testing(gpr_model, 
                 X_test=X, 
                 X_train=X_train, 
                 y_train=y_train,
                 y_test=y,
                 xlabel=xlabel,
                 ylabel=ylabel,
                 mean_absolute_error=mean_absolute_error,
                 out_dir=output_base_dir, out_fname='initial')
    opt_bounds = torch.stack([X.min(dim=0).values, X.max(dim=0).values])
    for _ in range(configs.get('n_optimization_steps')):
        updated_model = optimize_loop(model=gpr_model,
                                      loss=gpr_mll, 
                                      X_train=X_train, 
                                      y_train=y_train,
                                      X_test=X_test, 
                                      y_test=y_test,
                                      bounds=opt_bounds,
                                      xlabel=xlabel)
        gpr_model, gpr_mll = updated_model['model'], updated_model['loss']
        X_train, y_train = updated_model['X_train'], updated_model['y_train']
        X_test, y_test = updated_model['X_test'], updated_model['y_test']
        X_new, y_new = updated_model['X_new'], updated_model['y_new']
        mean_absolute_error = get_mean_absolute_error(X_test=X_test, 
                                                      y_test=y_test, 
                                                      gpr_model=gpr_model, 
                                                      y_mean=y_mean, 
                                                      y_scale=y_scale)
        maes.append(mean_absolute_error)
        y_test_means.append(float(torch.mean(y_test).numpy()))
        plot_testing(gpr_model, 
                     X_test=X, X_train=X_train, 
                     y_test=y, y_train=y_train,
                     xlabel=xlabel, ylabel=ylabel,
                     X_new=X_new, y_new=y_new, 
                     mean_absolute_error=mean_absolute_error,
                     out_dir=output_base_dir, out_fname='image'+str(_+1))
    plot_maes(maes, 
              out_dir=output_base_dir, 
              mae_scale=y_range, 
              mae_scale_label='Range',
              response_name=configs.get('response').upper())
    plot_y_test_means(y_test_means,
                      out_dir=output_base_dir, 
                      response_name=configs.get('response').upper())
        
    # plt.plot([_ for _ in range(configs.get('n_optimization_steps'))], max_val, 
    #          'go--', linewidth=2, markersize=12)
    # plt.fill_between([_ for _ in range(configs.get('n_optimization_steps'))], 
    #                  lower_confidence, upper_confidence, 
    #                  alpha=0.5, label = '95% Credibility')
    # plt.show()

if __name__ == "__main__":
    main()