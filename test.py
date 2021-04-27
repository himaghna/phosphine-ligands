# -*- coding: utf-8 -*-
""" Bayesian Optimization using GPR to find the optima of a synthetic dataset 
generated from a function
@uthor: Himaghna, 22nd November 2019

Attributes
----------
device : Pytorch object
    A torch.device is an object representing the device on which a 
    torch.Tensor is or will be allocated.
dtype: Pytorch object
    A torch.dtype is an object that represents the data type of a torch.Tensor.
y_mean: float
    The mean of the original dataset-set response around which all responses
    are centered. Set in get_observations() when generating initial points.
y_std: float
    The standard deviation of the original dataset-set response which is used
    to scale all responses. Set in get_observations() when generating 
    initial points.

"""


import matplotlib
import matplotlib.pyplot as plt
from numpy import random
import torch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
y_mean, y_std = None, None

def eval_true_func(x, add_noise=True):
    """ Get an evaluation of the underlying function with gaussian noise added


    Parameters
    ----------

    x: (sx1) Tensor 
        Point(s) at which the functional evaluation(s) is (are) carried out.
        Points are in [0, 1]
    add_noise: bool
        If Gaussian noise is to be added to the observation.
        Default is True.
    
    
    Returns
    -------
    func_value: (sx1) Tensor
        Value(s) of function at given x(s).

    
    """
    # function evaluation
    func_value = ((x-0.1) * (x-2.2) * (x-0.5) * torch.sin(x) * (x-2.5))
    # add 10% gaussian noise if desired
    noise = 0.1 * torch.rand_like(func_value) if add_noise is True else 0
    func_value += noise
    return func_value

def get_observations(n_samples=10, uniform_grid_in=None, **kwargs):
    """Get an n_sampleS and observations.


    Parameters
    ----------


    n_samples: int
    Number of initial training points.
    uniform_grid_in: tuple
        Grid bounds as (lower, upper). If passed a set of 
        n_samples distributed evenly in this space is returned. If None then
        samples are drawn from a N(0,1) distribution. Default is None
    **kwargs: key word argument 
        arguments to modify function behaviour. Some important ones:
        add_noise: Boolean: passed to eval_true_func()
    

    Returns
    -------


    synthetic_data: dict
        X: (n_samples x 1) Tensor
            Training Covariates.
        y: (n_samples x 1) Tensor
            Training Observations.
    
    
    """
    global y_mean
    global y_std
    # synthesize covariates
    if uniform_grid_in is None:
        # random samples
        X = torch.randn((n_samples, 1), dtype=dtype)
        X, _ = torch.sort(X, dim=0)
    else:
        X = torch.linspace(uniform_grid_in[0], uniform_grid_in[1], n_samples, \
            dtype=dtype)
        X = X.view((-1, 1))  # reshaping to (-1, 1)

    # get functional evaluation for training data
    y = eval_true_func(X)
    # Normalize y
    if y_mean is None or y_std is None:
        # set global variables if they are at None
        y_mean = y.mean()
        y_std = y.std()
    y = (y - y_mean)/ y_std
    synthetic_data = {
        'X': X,
        'y': y
    }
    return synthetic_data

def get_gpr_model(X, y, model=None):
    """Fit a gpr model to the data or update the model to new data.


    Parameters
    ----------


    X: (sx1) Tensor
        Covariates
    y: (sx1) Tensor
        Observations
    model: PyTorch SingleTaskGP model
        If model is passed, X and y are used to update it. 
        If None then model is trained on X and y. Default is None.

    
    Returns
    -------


    model: PyTorch SingleTaskGP model
        Trained or updated model. Returned in train mode.
    mll: PyTorch MarginalLogLikelihood object
        This is the loss used to train hyperparameters. Returned in train mode.


    """
    if model is None:
        # set up model
        print('X', X.shape)
        print('y', y.shape)
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

def plot_testing(model, *, X_train, y_train, X_test, y_test=None, X_new=None, \
    y_new=None):
    """Plot the performance of the model


    Parameters
    ----------
    model: PyTorch SingleTaskGP model
        Model to test using testing data and plot performance
    X_train: (sx1) Tensor
        Vector of inputs for training model.
    y_train: (sx1) Tensor
        Vector of outputs for training model.
    X_test: (sx1) Tensor
        Vector of inputs for testing model.
    y_test: (sx1) Tensor
        Vector of outputs for testing model. Default is None.
    X_new: (1x1) Tensor
        Single input representing new point that will be added to training set.
        Default is None.
    y_new: (1x1) Tensor
        Value of the functional evaluation at X_new.
        Default is None.


    """
    # Initialize plot
    font = {'size'   : 20}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 1.5
    matplotlib.rcParams['xtick.major.size'] = 8
    matplotlib.rcParams['xtick.major.width'] = 2
    matplotlib.rcParams['ytick.major.size'] = 8
    matplotlib.rcParams['ytick.major.width'] = 2
    fig, ax = plt.subplots(figsize=(12, 6))
    # set up model in eval mode
    model.eval();

    with torch.no_grad():
        # compute posterior
        posterior = model.posterior(X_test)
        # Get upper and lower confidence bounds (2 standard deviations from the mean)
        lower, upper = posterior.mvn.confidence_region()
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
        ax.scatter(X_train.cpu().numpy(), y_train.cpu().numpy(), \
            s=120, c= 'k', marker = '*', label = 'Initial Data')
         # Plot the new infill points as red stars
        if X_new is not None:    
            ax.scatter(X_new.cpu().numpy(), y_new.cpu().numpy(), \
                s=120, c='r', marker='*', label='Infill Data')
        
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.tight_layout()
    plt.show()

def plot_acq_func(acq_func, X_test, X_train, X_new=None):
    # compute acquisition function values at test_X
    test_acq_val = acq_func(X_test.view((X_test.shape[0], 1, 1)))
    train_acq_val = acq_func(X_train.view((X_train.shape[0], 1, 1)))

    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 6))
    with torch.no_grad():
        ax.plot(X_test.cpu().numpy(), test_acq_val.detach(), 'b-', \
            label='Acquisition (EI)')
        # Plot training points as black stars
        ax.scatter(X_train.cpu().numpy(), train_acq_val.detach(), s = 120, \
            c='k', marker = '*', label = 'Initial Data')
         # Plot the new infill points as red stars
        if X_new is not None: 
            new_acq_val = acq_func(X_new.view((X_new.shape[0], 1, 1)))
            ax.scatter(X_new.cpu().numpy(), new_acq_val.detach(), \
                s=120, c='r', marker = '*', label = 'Infill Data')
    
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-2,2) )
    ax.set_xlabel('x')
    ax.set_ylabel(r'$ \alpha$')    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

def optimize_loop(model, loss, X_train, y_train, X_test, y_test, bounds):
    best_value = y_train.min()
    acq_func = ExpectedImprovement(model, best_f= best_value, maximize=False)
    X_new, acq_value = optimize_acqf(acq_func, bounds=bounds, q=1, \
        num_restarts=100, raw_samples=100)
    X_new = X_new.view((1,1))
    y_new = (eval_true_func(X_new) - y_mean) / y_std
    # concatenate new points to training set
    X_train_new = torch.cat((X_train, X_new))
    y_train_new = torch.cat((y_train, y_new))
    plot_acq_func(acq_func, X_test=X_test, X_train=X_train_new, X_new=X_new)
    # condition model on new observation
    gpr_model, gpr_mll = get_gpr_model(X_new, y_new, model=model)
    # plot model performance
    plot_testing(gpr_model, X_test=X_test, X_train=X_train_new, \
        y_train=y_train_new, y_test=y_test, X_new=X_new, y_new=y_new)
    return gpr_model, gpr_mll, X_train_new, y_train_new

def main(): 
    # get some initial training data
    lower_x_limit = -10.
    upper_x_limit = 10.
    training_data = get_observations(n_samples=10, uniform_grid_in=(int(lower_x_limit), int(upper_x_limit)))
    X_train = training_data['X']
    y_train = training_data['y']
    # set up GPR model
    gpr_model, gpr_mll = get_gpr_model(X=X_train, y=y_train)

    # get some testing data
    testing_data = get_observations(n_samples=1000, uniform_grid_in=(int(lower_x_limit), int(upper_x_limit)))
    
    # plot model performance
    plot_testing(gpr_model, X_test=testing_data['X'], X_train=X_train, \
        y_train=y_train, y_test=testing_data['y'])
    
    # do some optimization!
    N_OPT_STEPS = 10
    opt_bounds = torch.tensor([[lower_x_limit], [upper_x_limit]], dtype=dtype)
    for _ in range(N_OPT_STEPS):
        gpr_model, gpr_mll, X_train, y_train = optimize_loop(model=gpr_model, \
            loss=gpr_mll, X_train=X_train, y_train=y_train, \
                X_test=testing_data['X'], y_test=testing_data['y'], \
                    bounds=opt_bounds)
    

if __name__ == '__main__':
    main()


        

