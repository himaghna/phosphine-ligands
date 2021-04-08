import numpy as np
import torch

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

def tensor_elements_split(inp_tensor, to_pop, device=None, is_index=True):
    """
    Pop elements from an input tensor
    Params ::
    inp_tensor: tensor: Input Tensor
    to_pop: array array like: 
        collection of indexes or elements to pop from inp_tensor
    device: Torch.Device
        Device to store the tensors.
    is_index: Boolean: if set to True, to_pop is treated as indices. If False
        to_pop is treated as list of elements. Default is True
    Return ::
    Tuple(popped_tensor, popped_elements)
        popped_tensor: Tensor of type inp_tensor: Input tensor with the 
            popped elements removed
        popped_elements: Tensor of type inp_tensor: Tensor of popped rows
    
    """
    if device is None:
        device = torch.device("cpu")
    if is_index is True:
        idx_to_keep = torch.tensor([id for id in range(inp_tensor.size(0)) 
                                        if id not in to_pop], 
                                    device=device, dtype=torch.long)
        if not torch.is_tensor(to_pop):
            to_pop = torch.tensor(to_pop, device=device, dtype=torch.long)
        popped_elements = inp_tensor.index_select(0, to_pop)
        popped_tensor = inp_tensor.index_select(0, idx_to_keep)

    else:
        raise NotImplementedError()
    return (popped_tensor, popped_elements)

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

def normalize_tensor(in_tensor, dim=0, mean_=None, std_deviation_=None):
    """Normalize a vector

    Parameters
    ----------
    in_tensor: torch.Tensor
        Tensor to normalize.
    dim: int
        Dimension to normalize along. Only needed if the tensors own statistics
        is used to normalize it (mean_ and std_deviation_ not supplied). 
        Default is 0.
    mean_: torch.Tensor
        Mean used to center the Tensor. Not needed if the tensors own statistics
        is used to normalize it.
    
    std_deviation_: torch.Tensor
        Standard deviation used to scale the Tensor. Not needed if the tensors 
        own statistics is used to normalize it.
    
    Returns
    -------
    dict
        'normalized': transformed tensor,
        'std':
        'mean':
    
    """  
    if mean_ is None or std_deviation_ is None:
        print('Normalizing vector using its own statistics')
        mean_ = in_tensor.mean(dim=dim)
        std_deviation_ = in_tensor.std(dim=dim)
    return {
        'normalized': (in_tensor - mean_) / std_deviation_,
        'std': std_deviation_,
        'mean': mean_
    }


def scaleup_tensor(in_tensor, mean_, std_deviation_):
    """Scaleup or "un-normalize" a tensor

    Parameters
    ----------
    in_tensor: torch.Tensor
        Tensor to normalize.

    mean_: torch.Tensor
        Mean used to re-center the Tensor. 
    
    std_deviation_: torch.Tensor
        Standard deviation used to re-scale the Tensor.
    Returns
    -------
    torch.Tensor
        Scaledup Tensor.

    """
    return in_tensor * std_deviation_ + mean_



def train_test_split(X, train_fraction, random_seed, y=None, device=None):
    train_size = int(train_fraction * X.shape[0])              
    
    np.random.seed(random_seed)
    initial_idx = list(np.random.choice(X.shape[0], 
                       train_size,
                       replace=False))
    X_test, X_train = tensor_elements_split(X, to_pop=initial_idx, device=device)
    if y is not None:
        y_test, y_train = tensor_elements_split(y, 
                                                to_pop=initial_idx, 
                                                device=device)
        return X_train, X_test, y_train, y_test
    else:
        return X_train, X_test

def move_test_to_train(training_data, test_data, id_of_point_to_move):
    """ Move a point from test to training dataset
    
    Parameters
    ---------
    training_data: tuple
        (X_train, y_train).
    test_data: tuple
        (X_test, y_test)
    id_of_point_to_move: int
        Index of point to move from test to trainind set. 
        Index is in reference to test data before operation.
    
    Returns
    -------
    tuple
        (X_train_new, y_train_new), (X_test_new, y_test_new)

    """
    (X_train, y_train) = training_data
    (X_test, y_test) = test_data 
    X_test_new, X_new = tensor_elements_split(inp_tensor=X_test,
                                              to_pop=id_of_point_to_move)
    y_test_new, y_new = tensor_elements_split(inp_tensor=y_test,
                                              to_pop=id_of_point_to_move)
    X_train_new = torch.cat((X_train, X_new))
    y_train_new = torch.cat((y_train, y_new))
    return (X_train_new, y_train_new), (X_test_new, y_test_new)

def get_new_point_to_acquire(acq_vals, test_data=None):
    max_acqf_id = acq_vals.argmax()
    if test_data is not None:
        (X_test, y_test) = test_data
        new_point_X = X_test.index_select(0, max_acqf_id)
        new_point_y = y_test.index_select(0, max_acqf_id)
        return (new_point_X, new_point_y), max_acqf_id
    else:    
        return max_acqf_id


def detach_tensor_to_numpy(in_tensor):
    if torch.is_tensor(in_tensor):
        return in_tensor.detach.numpy()
    return in_tensor

    
