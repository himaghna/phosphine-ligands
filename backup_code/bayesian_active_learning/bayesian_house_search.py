"""
@uthor: Himaghna, 22nd November 2019
Description: Perform Bayesian Optimization on the California data set
"""


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import torch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
import pandas as pd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float64


def sort_tensor(inp_tensor, sort_column_id):
    """
    Sort a tensor according the contents of a column
    Params ::
    inp_tensor: Tensor: Tensor to be sorted
    sort_column_id: int: index of column used for sorting
    Return ::
    (out_tensor, idx): Tuple: (Sorted Tensor, idx used for sorting)
    """
    # extract the column of inp_tensor corresponding to sort_column_id
    sort_column = inp_tensor[:, sort_column_id] 
    _, idx = sort_column.sort()
    # sort X and y so thaat x_dim is sorted
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
        pass
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
        model = model.condition_on_observations(X, y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(X)
    # begin training
    model.train();
    mll.train();
    fit_gpytorch_model(mll);
    return model, mll

def plot_testing(model, X_test, X_train, y_train, y_test=None, X_new=None, y_new=None):
    '''
    Test the surrogate model with model, test_X and new_X
    '''

    # Initialize plot
    font = {'size'   : 20}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 1.5
    matplotlib.rcParams['xtick.major.size'] = 8
    matplotlib.rcParams['xtick.major.width'] = 2
    matplotlib.rcParams['ytick.major.size'] = 8
    matplotlib.rcParams['ytick.major.width'] = 2
    hist_fig, ax = plt.subplots(figsize=(12, 6))
    # set up model in eval mode
    model.eval();
    x_dim = 1 # Visualize data along this dimension
    X_test, idx = sort_tensor(X_test, sort_column_id=x_dim)
    y_test = y_test.index_select(0, idx)

    with torch.no_grad():
        posterior = model.posterior(X_test)
        # Get upper and lower confidence bounds (2 std from the mean)
        lower, upper = posterior.mvn.confidence_region()
        # visualize along only one x direction
        X_test = X_test[:, x_dim]
        # sort X and y for prettiness
        # Plot the ground truth test_Y if provided
        ax.plot(X_test.cpu().numpy(), y_test.cpu().numpy(), \
            'k--', label='Objective f(x)')
        # Plot posterior means as blue line
        ax.plot(X_test.cpu().numpy(), posterior.mean.cpu().numpy(), \
            'b', label='Posterior Mean')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(X_test.cpu().numpy().squeeze(), lower.cpu().numpy(), \
            upper.cpu().numpy(), alpha=0.5, label = '95% Credibility')
        
        # Plot training points as black stars
        #ax.scatter(X_train[:, x_dim].cpu().numpy(), y_train.cpu().numpy(), \
         #   s=120, c= 'k', marker = '*', label = 'Initial Data')
         # Plot the new infill points as red stars
        if X_new is not None:    
            ax.scatter(X_new[:, x_dim].cpu().numpy(), y_new.cpu().numpy(), \
                s=120, c='r', marker='*', label='Infill Data')
        
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

def plot_acq_func(acq_func, X_test, X_train, X_new=None, x_visualize_dim=1):
    # compute acquisition function values at test_X
    test_acq_val = acq_func(X_test.view((X_test.shape[0], 1, X_test.shape[1])))
    # Initialize plot
    hist_fig, ax = plt.subplots(figsize=(12, 6))
    with torch.no_grad():
        ax.scatter(X_test[:, x_visualize_dim].cpu().numpy(), 
        test_acq_val.cpu().detach(), c='blue', s=1.2, \
            alpha=0.7, label='Acquisition (EI)')
         # Plot the new infill points as red stars
        if X_new is not None: 
            new_acq_val = acq_func(X_new.view(
                (X_new.shape[0], 1, X_new.shape[1])))
            ax.scatter(X_new[:, x_visualize_dim].cpu().numpy(), \
                new_acq_val.cpu().detach(), \
                s=120, c='r', marker='*', label='Infill Data')
    
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-2,2) )
    ax.set_xlabel('x')
    ax.set_ylabel(r'$ \alpha$')    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

def optimize_loop(model, loss, X_train, y_train, X_test, y_test, bounds):
    best_value = y_train.max()
    acq_func = ExpectedImprovement(model, best_f=best_value, maximize=True)
    acq_vals = acq_func(X_test.view((X_test.shape[0], 1, X_test.shape[1])))
    # this point has maximum acquisition value and will be added
    max_acqf_id = acq_vals.argmax()
    # get the new testing point from X_test
    X_test, X_new = tensor_pop(inp_tensor=X_test,
        to_pop=max_acqf_id.cpu().numpy())
    y_test, y_new = tensor_pop(inp_tensor=y_test,
        to_pop=max_acqf_id.cpu().numpy())
    # concatenate new points to training set
    X_train = torch.cat((X_train, X_new))
    y_train = torch.cat((y_train, y_new))
    # plot acq function
    #plot_acq_func(acq_func, X_test=X_test, X_train=X_train, X_new=X_new)
    # condition model on new observation
    gpr_model, gpr_mll = get_gpr_model(X_new, y_new, model=model)
    # plot model performance
    #plot_testing(gpr_model, X_test=X_test, X_train=X_train, \
    #    y_train=y_train, y_test=y_test, X_new=X_new, y_new=y_new)
    return gpr_model, gpr_mll, X_train, y_train, X_test, y_test

def explore_dataset(dataset, data_size=None):
    """
    Visualize the data set 
    Params ::
    dataset: SKLEARN class
    data_size: int: if provided, these many elements of data set considered
    Returns ::
    None
    """
    dataset_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    if data_size is not None:
        dataset_df = dataset_df[:data_size]
    print(dataset_df.describe())

    # plot histogram and scatter of all descriptors
    hist_fig, hist_axs = plt.subplots(4, 2)
    scatt_fig, scatt_axs = plt.subplots(4, 2)
    cmap = matplotlib.cm.get_cmap('magma')
    colors = [cmap(_/8) for _ in range(8)]  
    for id in range(len(dataset.feature_names)):
        hist_axs[int(id / 2), \
            int(id % 2)].hist(dataset_df[dataset.feature_names[id]].values,\
                color=colors[id], bins=20)
        scatt_axs[int(id / 2), \
            int(id % 2)].scatter([_ for _ in \
                range(len(dataset_df[dataset.feature_names[id]].values))], \
                    dataset_df[dataset.feature_names[id]].values,\
                c=colors[id], s=0.1, alpha=0.5)
        hist_axs[int(id / 2), int(id % 2)].set_title(dataset.feature_names[id])
        scatt_axs[int(id / 2), int(id % 2)].set_title(dataset.feature_names[id])
    for ax in scatt_axs.flat:
        ax.set(xlabel='sample index', ylabel='Feature Value')
    hist_fig.tight_layout()
    scatt_fig.tight_layout()
    plt.show()

    # look at response
    response_df = pd.DataFrame(dataset.target)
    if data_size is not None:
        response_df = response_df[:data_size]
    print(response_df.describe())
    plt.hist(response_df.values)
    plt.show()
    
def main():
    # import the data
    dataset = datasets.fetch_california_housing()
    X = torch.tensor(dataset.data, device=device)
    y = torch.tensor(dataset.target.reshape(-1,1), device=device)
    feature_names = dataset.feature_names
    #explore_dataset(dataset) # explore dataset
    # normalize X and y
    y_scale = y.std(dim=0)
    y_mean = y.mean(dim=0)
    X = (X - X.mean(dim=0)) / X.std(dim=0)
    y = (y - y_mean) / y_scale
    # 10% of the data is used as initial training points. The rest are test data
    initial_data_size = 10        #int(0.001* X.shape[0])
    initial_idx = list(np.random.choice(X.shape[0], initial_data_size, \
        replace=False))
    X_test, X_train = tensor_pop(X, to_pop=initial_idx)
    y_test, y_train = tensor_pop(y, to_pop=initial_idx)

    # set up GPR model
    gpr_model, gpr_mll = get_gpr_model(X=X_train, y=y_train)
    
    # plot model performance
    plot_testing(gpr_model, X_test=X_test, X_train=X_train, \
        y_train=y_train, y_test=y_test)

    # do some optimization!
    N_OPT_STEPS = 10
    opt_bounds = torch.stack([X.min(dim=0).values, X.max(dim=0).values])
    max_val, upper_confidence, lower_confidence = [], [], []
    for _ in range(N_OPT_STEPS):
                # get the point which has the maximum posterior and the variance to it
        gpr_model.eval();
        posterior = gpr_model.posterior(X_train)
        lower, upper = posterior.mvn.confidence_region()
        max_posterior, index = posterior.mean.max(dim=0)
        max_val.append(float(max_posterior * y_scale + y_mean))
        upper_confidence.append(float(upper[index] * y_scale + y_mean))
        lower_confidence.append(float(lower[index] * y_scale + y_mean))

        gpr_model, gpr_mll, X_train, y_train, X_test, y_test = optimize_loop(
            model=gpr_model, \
            loss=gpr_mll, X_train=X_train, y_train=y_train, \
                X_test=X_test, y_test=y_test, \
                    bounds=opt_bounds)

    plt.plot([_ for _ in range(N_OPT_STEPS)], max_val, \
        'go--', linewidth=2, markersize=12)
    plt.fill_between([_ for _ in range(N_OPT_STEPS)], lower_confidence, \
            upper_confidence, alpha=0.5, label = '95% Credibility')
    plt.show()


if __name__ == "__main__":
    main()