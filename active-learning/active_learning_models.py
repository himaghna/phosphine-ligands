import torch
from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

from tensor_ops import tensor_elements_split


def get_new_points_acq_func_vals(model, 
                                 acq_fn_label, 
                                 new_points, 
                                 best_response,
                                 acq_fn_hyperparams=None):
    if acq_fn_label == 'expected_improvement':
        acq_func = ExpectedImprovement(model, best_f=best_response, maximize=True)
    elif acq_fn_label == 'ucb':
        hyperparams = {'beta': 2}
        if acq_fn_hyperparams is not None:
            hyperparams.update(acq_fn_hyperparams)
        acq_func = UpperConfidenceBound(model, **hyperparams)
    else:
        raise NotImplementedError(f'acq_fn_label {acq_fn_label} does not ' 
                                   'match implemented types')
    acq_vals = acq_func(new_points.view((new_points.shape[0], 
                                         1, 
                                         new_points.shape[1])))
    return acq_vals


def train_gpr_model(X, y, model=None):
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
