"""
@uthor: Himaghna, 23rd September 2019
Description: Fit various models to predict E/Zs on the data-set
"""


import os.path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import data_processing
from helper_functions import plot_parity

processed_data = data_processing.main()
X, y = processed_data['X'], processed_data['y'].reshape(-1,1)
descriptor_names = processed_data['descriptor_names']

#family_idx = processed_data['family_int']

# Mask Family
family_response_threshold = 4
# only select ligands with response > threshold
idx_mask = (y > family_response_threshold) 
X, y = X[idx_mask], y[idx_mask] 
# pyplot parameters
plt.rcParams['svg.fonttype'] = 'none'
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)


def do_PCR(variance_needed=0.90):
    """
    Do PCR using top n_eigenvectors of the data matrix
    Params ::
    n_eigenvectors: int: Number of eigenvectors to retain. If set to None all
        are used. Defult None
    """

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)
    X_std_train = x_scaler.fit_transform(X_train)
    X_std_test = x_scaler.transform(X_test)
    y_std_train = y_scaler.fit_transform(y_train)
    y_std_test = y_scaler.transform(y_test)
    y_sigma = y_scaler.scale_
    pca = PCA()
    pca.fit(X_std_test)
    def get_cumulative(in_list):
        """

        :param in_list: input list of floats
        :return: returns a cumulative values list
        """
        out_list = list()
        for key, value in enumerate(in_list):
            assert isinstance(value, int) or isinstance(value, float)
            try:
                new_cumulative = out_list[key-1] + value
            except IndexError:
                # first element
                new_cumulative = value
            out_list.append(new_cumulative)
        return out_list
    cum_var = get_cumulative([i for i in pca.explained_variance_ratio_])

    for key, value in enumerate(cum_var):
        if value >= variance_needed:
            n_eigenvectors_needed = (key+1)
            print('{} explained by {} eigen-vectors'.format(value,
                                                            n_eigenvectors_needed))
            break
    
    # do reduced PCA
    pca_reduced = PCA(n_components=n_eigenvectors_needed)
    X_std_train = pca_reduced.fit_transform(X_std_train)
    X_std_test = pca_reduced.transform(X_std_test)

    # do regression

    lm = LinearRegression()
    lm.fit(X_std_train, y_std_train)
    y_predict = [(_ * y_sigma) + y_scaler.mean_ for _ in lm.predict(X_std_test)]
    y_test =[(_ * y_sigma) + y_scaler.mean_ for _ in y_std_test]
    print('Mean Absolute Error: ', mean_absolute_error(y_true=y_test, \
        y_pred=y_predict))
    print('R2 of training data: ', lm.score(X_std_train, y_std_train))
    plot_parity(x=y_test, y=y_predict, xlabel='True E/Z Ratio', \
        ylabel='Predicted E/Z Ratio')



def do_LASSO(cv=10):
    """
    Do LASSO on the data-set
    Params ::
    cv: int: folds of craoss-validation to do. Default 10
    Returns ::
    None
    """
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.60, random_state=23)
    X_std_train = x_scaler.fit_transform(X_train)
    X_std_test = x_scaler.transform(X_test)
    y_std_train = y_scaler.fit_transform(y_train)
    y_std_test = y_scaler.transform(y_test)
    y_sigma = y_scaler.scale_

    lasso = LassoCV(cv=cv)
    lasso.fit(X_std_train, y_std_train)
    y_predict = [(_ * y_sigma) + y_scaler.mean_ for _ in lasso.predict(X_std_test)]
    print('Mean Absolute Error: ', mean_absolute_error(y_true=y_test, \
        y_pred=y_predict))
    print('R2 of training data: ', lasso.score(X_std_train, y_std_train))
    plot_parity(x=y_test, y=y_predict, xlabel='True E/Z Ratio', \
        ylabel='Predicted E/Z Ratio')
    
    
    


def main():
    #do_PCR(variance_needed=0.90)
    do_LASSO(cv=10)

if __name__ == '__main__':
    main()
