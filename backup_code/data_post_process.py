"""
@uthor: Himaghna, 1st October 2019
Description: Methods to post-process input and output
"""


from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split


def bin_to_families(in_column, n_bins=None, bin_upper_bounds=None):
    """
    Take in a numerical input and bin it into families
    Params ::
    in_column: np array or list of size s: input vector containing observations
        of a continuous variable
    n_bins : int: Number of bins to create. Data is binned into families each of
        size (maxval - minval)/ n_bins. Default None
    bin_upper_bounds: list[floats]: Upper bounds for bins. Only used if 
       n_bins is not set. Used for creating custom bins.
    bin_upper_bounds is inclusive i.e. UPPER BOUNDS ARE INCLUDED IN THE BIN
    
    Returns ::
    out_families: s x n_bins np array: one hot encoded  bin membership
    out_families_ordinal: s x 1 np array: binned response 
        with each response being being bin id for that row
    """
    in_column = np.array(in_column).reshape(-1, 1)
    out_families, out_families_ordinal = [], []
    min_element, max_element = np.amin(in_column), np.amax(in_column)
    if n_bins is not None:
        bin_size = (max_element - min_element) / n_bins
        # create upper bin limits assuming max_element is the last bound
        bin_upper_bounds = [min_element + bin_size * bin_id \
                              for bin_id in range(1, n_bins)]
    else:
        n_bins = len(bin_upper_bounds) + 1
    
    # bin
    for observation in in_column:
        one_hot_vector = [0] * n_bins
        def encode_one_hot():
            for bin_id, bin_upper_bound in enumerate(bin_upper_bounds):
                if observation <= bin_upper_bound:
                    one_hot_vector[bin_id] = 1
                    return one_hot_vector, bin_id
            one_hot_vector[-1] = 1
            return one_hot_vector, len(bin_upper_bounds)
        one_hot_vector, bin_id = encode_one_hot()
        out_families.append(one_hot_vector)
        out_families_ordinal.append(bin_id)

    out_families = np.array(out_families).flatten()
    out_families_ordinal = np.array(out_families_ordinal).flatten()

    return out_families, out_families_ordinal


def get_family_membership_idx(in_column):
    """
    Return a dictionary of idx based on family membership as encoded in the 
    ordinal observtion vector in_column
    Params ::
    in_column: s x 1 np array or array-like: Ordinal responses encoding class
        membership
    Returns ::
    idx: dict: key -> class number, value = List(indices of class in in_column)
    """
    idx = defaultdict(list)

    for id, observation in enumerate(in_column):
        idx[int(observation)].append(id)
    return idx


def get_train_test_idx(class_membership_dict, *, train_size=0.8, random_state=None):
    """
    Balanced train-test split from a multi-class response set
    Params ::
    class_membership_dict: dict: dictionary containing class id as keys and
        idx of samples belonging to that class as vals
    train_size: float: portion of the data_set to be used for training.
         Default 0.8
    random_state = seed for the split. Default is None

    Return ::
    train_test_idx: dict: keys ['train', 'test']. 
                    vals idx for train and test respectively
    """
    train_test_idx = defaultdict(list)
    for key in class_membership_dict:
        train, test = train_test_split(class_membership_dict[key], \
            train_size=train_size, random_state=random_state)
        train_test_idx['train'].extend(train)
        train_test_idx['test'].extend(test)
    return train_test_idx










     