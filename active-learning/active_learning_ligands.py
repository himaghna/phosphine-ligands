""" Active learning of ligands"""
from argparse import ArgumentParser
import pickle
from os import mkdir
from os.path import isdir, join

import matplotlib
import matplotlib.pyplot as plt
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


def sort_tensor(inp_tensor, sort_column_id):
    """
    Sort a tensor according the contents of a column
    Params ::
    inp_tensor: Tensor: Tensor to be sorted
    sort_column_id: int: index of column used for sorting
    Return ::
    (out_tensor, idx): Tuple: (Sorted Tensor, idx used for sorting)
    """
    sort_column = inp_tensor[:, sort_column_id] 
    _, idx = sort_column.sort()
    out_tensor = inp_tensor.index_select(0, idx)
    return (out_tensor, idx)

def tensor_pop(inp_tensor, to_pop, is_index=True):
    """
    Pop elements from an input tensor
    Params ::
    inp_tensor: tensor: Input Tensor
    to_pop: array array like: 
        collection of indexes or elements to pop from inp_tensor
    is_index: Boolean: if set to True, to_pop is treated as indices. If False
        to_pop is treated as list of elements. Default is True
    Return ::
    Tuple(out_tensor, popped_elements):
        out_tensor: Tensor of type inp_tensor: Input tensor with the 
            popped elements removed
        popped_elements: Tensor of type inp_tensor: Tensor of popped rows
    """
    if is_index is True:
        idx_to_keep = torch.tensor([id for id in range(inp_tensor.size(0)) \
            if id not in to_pop], device=device, dtype=torch.long)
        to_pop = torch.tensor(to_pop, device=device, dtype=torch.long)
        popped_elements = inp_tensor.index_select(0, to_pop)
        out_tensor = inp_tensor.index_select(0, idx_to_keep)

    else:
        raise NotImplementedError()
    return (out_tensor, popped_elements)

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

def plot_testing(model, 
                 X_test, X_train, 
                 y_train, 
                 xlabel,
                 ylabel,
                 y_test=None, mean_absolute_error=None,
                 X_new=None, y_new=None, out_dir=None, out_fname=None):
    font = {'size'   : 20}
    matplotlib.rc('font', **font)
    plt.rcParams['svg.fonttype'] = 'none'
    matplotlib.rcParams['axes.linewidth'] = 1.5
    matplotlib.rcParams['xtick.major.size'] = 8
    matplotlib.rcParams['xtick.major.width'] = 2
    matplotlib.rcParams['ytick.major.size'] = 8
    matplotlib.rcParams['ytick.major.width'] = 2
    hist_fig, ax = plt.subplots(figsize=(12, 6))
    model.eval();
    X_test, idx = sort_tensor(X_test, sort_column_id=VISUALIZATION_DIM)
    y_test = y_test.index_select(0, idx)

    with torch.no_grad():
        posterior = model.posterior(X_test)
        posterior_mean = posterior.mean.cpu().numpy()
        lower, upper = posterior.mvn.confidence_region()
        X_test = X_test[:, VISUALIZATION_DIM]
        ax.plot(X_test.cpu().numpy(), y_test.cpu().numpy(), 
                'lightcoral', label=f'True {ylabel}')
        ax.plot(X_test.cpu().numpy(), posterior_mean,
                'b', label='Posterior Mean')
        ax.fill_between(X_test.cpu().numpy().squeeze(), 
                        lower.cpu().numpy(), upper.cpu().numpy(), 
                        alpha=0.5, label='95% Credibility')   
        ax.scatter(X_train[:, VISUALIZATION_DIM].cpu().numpy(), 
                   y_train.cpu().numpy(),
                   s=150, c='m', marker='*', label='Training Data')

        if X_new is not None:    
            ax.scatter(X_new[:, VISUALIZATION_DIM].cpu().numpy(), 
                       y_new.cpu().numpy(),
                       s=150, c='r', marker='*', label='Infill Data')
        
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    if mean_absolute_error is not None:
        plt.text(0.7,  0.9, 'MAE: {:.2f}'.format(mean_absolute_error), 
                transform=ax.transAxes, fontsize=16)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    if INTERACTIVE:
        plt.show()
    else:
        if not isdir(out_dir):
            mkdir(out_dir)
        subdir = join(out_dir, 'test-plots')
        if not isdir(subdir):
            mkdir(subdir)
        out_fpath = join(subdir, out_fname+'.svg')
        plt.savefig(out_fpath)
        plt.clf()

def get_mean_absolute_error(X_test, y_test, gpr_model, y_mean=None, y_scale=None):
    """
    y_true: torch Tensor
        True response.
    y_predicted: torch Tensor
        Predicted response.
    
    Returns
    -------
    float
        Mean Absolute Error

    """
    with torch.no_grad():
        gpr_model.eval();
        posterior = gpr_model.posterior(X_test)
        abs_error = torch.abs(y_test - posterior.mean)
        if y_mean is not None and y_scale is not None:
            abs_error = abs_error * y_scale + y_mean
    return float(torch.mean(abs_error, axis=0).cpu().numpy())

def plot_acq_func(acq_func, X_test, X_train, xlabel, 
                  X_new=None, out_dir=None, out_fname=None):
    test_acq_val = acq_func(X_test.view((X_test.shape[0], 1, X_test.shape[1])))
    hist_fig, ax = plt.subplots(figsize=(12, 6))
    with torch.no_grad():
        ax.scatter(X_test[:, VISUALIZATION_DIM].cpu().numpy(), 
                   test_acq_val.cpu().detach(), 
                   c='blue', s=120, alpha=0.7, label='Acquisition (EI)')
        if X_new is not None: 
            new_acq_val = acq_func(X_new.view((X_new.shape[0], 
                                               1, 
                                               X_new.shape[1])))
            ax.scatter(X_new[:, VISUALIZATION_DIM].cpu().numpy(),
                       new_acq_val.cpu().detach(),
                       s=120, c='r', marker='*', label='Infill Data')
    
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-2,2) )
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(r'$ \alpha$', fontsize=20)    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    if INTERACTIVE:
        plt.show()
    else:
        if not isdir(out_dir):
            mkdir(out_dir)
        subdir = join(out_dir, 'acq-plots')
        if not isdir(subdir):
            mkdir(subdir)
        out_fpath = join(subdir, out_fname+'.svg')
        plt.savefig(out_fpath)
        plt.clf()

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

def plot_maes(maes, out_dir):
    """Plot the Mean Absolute Errors

    Parameters
    ----------
    maes: List(float)

    """
    plt.rcParams['svg.fonttype'] = 'none'
    plt.plot([0.1 * i for i in range(len(maes))], maes,
             marker='s', markerfacecolor='m', markeredgecolor='black', 
             c='m', markersize=30,
             markeredgewidth=2)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel('Optimization Step')
    plt.ylabel('Test MAE')
    if INTERACTIVE:
        plt.show()
    else:
        if not isdir(out_dir):
            mkdir(out_dir)
        out_fpath = join(out_dir, 'mae-plot.svg')
        plt.savefig(out_fpath)
        plt.clf()
    
def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='config yaml')
    args = parser.parse_args()
    with open(args.config, "r") as fp:
        configs = yaml.load(fp, Loader=yaml.FullLoader)

    X = torch.from_numpy(pickle.load(open(configs.get('X'), "rb"))).type(dtype)
    y = torch.from_numpy(pickle.load(open(configs.get('y'), "rb"))
                        ).type(dtype).reshape(-1, 1)
    descriptor_names = pickle.load(open(configs.get('descriptor_names'), "rb"))
    output_base_dir = configs.get('output_dir')
    ylabel = configs.get('response')
    xlabel = descriptor_names[VISUALIZATION_DIM]
    y_scale = y.std(dim=0)
    y_mean = y.mean(dim=0)
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0)
    X = (X - X_mean) / X_std
    y = (y - y_mean) / y_scale
    proportion_of_dataset_for_seeding = float(
                                  configs.get('proportion_of_dataset_for_seeding'))
    initial_data_size = int(proportion_of_dataset_for_seeding * X.shape[0])
    print(f'Using {initial_data_size} / {X.shape[0]} points for initial seeding')
    np.random.seed(int(configs.get('random_seed')))
    initial_idx = list(np.random.choice(X.shape[0], 
                       initial_data_size,
                       replace=False))
    X_test, X_train = tensor_pop(X, to_pop=initial_idx)
    y_test, y_train = tensor_pop(y, to_pop=initial_idx)
    maes = []
    
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
        plot_testing(gpr_model, 
                     X_test=X, X_train=X_train, 
                     y_test=y, y_train=y_train,
                     xlabel=xlabel, ylabel=ylabel,
                     X_new=X_new, y_new=y_new, 
                     mean_absolute_error=mean_absolute_error,
                     out_dir=output_base_dir, out_fname='image'+str(_+1))
        
    plot_maes(maes, out_dir=output_base_dir)
        
    # plt.plot([_ for _ in range(configs.get('n_optimization_steps'))], max_val, 
    #          'go--', linewidth=2, markersize=12)
    # plt.fill_between([_ for _ in range(configs.get('n_optimization_steps'))], 
    #                  lower_confidence, upper_confidence, 
    #                  alpha=0.5, label = '95% Credibility')
    # plt.show()

if __name__ == "__main__":
    main()