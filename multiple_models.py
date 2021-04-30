from argparse import ArgumentParser
import pickle

import forestci as fci
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from plotting_functions import plot_parity

def get_confidence_interval_from_std(std):
    return 1.96 * std

def get_forest(X_train, y_train, **hyperparams):
    rf = RandomForestRegressor(**hyperparams)
    rf.fit(X_train, y_train)
    return rf


def get_forest_conf_interval(rf_model, X_test, X_train):
    prediction_variance = fci.random_forest_error(rf_model, X_train, X_test)
    cli_95 = get_confidence_interval_from_std(np.sqrt(prediction_variance))
    return cli_95


def get_gpr(X_train, y_train, **hyperparams):
    kernel = Matern()
    gpr = GaussianProcessRegressor(kernel=kernel, **hyperparams)
    gpr.fit(X_train, y_train)
    return gpr


def  predict_on_test_data(model_, X_test, model_type, X_train=None):
    if model_type == 'gpr':
        y_pred, y_std = model_.predict(X_test, return_std=True)
        cli_95 = get_confidence_interval_from_std(y_std)
    elif model_type=='rf':
        y_pred = model_.predict(X_test)
        cli_95 = get_forest_conf_interval(rf_model=model_, 
                                          X_test=X_test, 
                                          X_train=X_train)
    else:
        raise NotImplementedError(f'{model_type} not implemented')
    return y_pred, cli_95


def analyze_on_test_data(y_test, y_test_pred, cli_95=None):
    plot_settings = {'c': '#ff5e78',
                     's': 10,} 
    error_plot_settings = {'ecolor': '#ff5e78',
                           'elinewidth': 0.5,
                           'alpha': 0.4,
                           'fmt': 'o'
                           }       
    plt.rcParams['svg.fonttype'] = 'none'
    plt.scatter(range(len(y_test_pred)), y_test_pred, **plot_settings, alpha=0.6)
    plt.errorbar(range(len(y_test_pred)), 
                 y_test_pred, 
                 yerr=cli_95, 
                 **error_plot_settings)
    plt.xlabel('Sample Index', fontsize=20)
    plt.ylabel('Predicted Response', fontsize=20)
    plt.show()

    if y_test is not None:
        mae = mean_absolute_error(y_true=y_test, y_pred=y_test_pred)
        r2 = r2_score(y_true=y_test, y_pred=y_test_pred)
        plot_parity(x=y_test, 
                    y=y_test_pred, 
                    xlabel='True response',
                    ylabel='Predicted response',
                    **plot_settings, 
                    show_plot=False,
                    text='MAE: {:.2f} R2: {:.2f}'.format(mae, r2),
                    text_x=0.1,
                    text_y=0.9)
        plt.errorbar(y_test, y_test_pred, yerr=cli_95, **error_plot_settings)
        plt.show()


if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument('-xt', help='Path of X.p')
    parser.add_argument('-yt', help='Path of y.p')
    parser.add_argument('-xv', required=False, help='Path of X_test.p')
    parser.add_argument('-yv', required=False, help='Path of y_test.p')
    parser.add_argument('-trf', '--training_fraction',
                        required=False, 
                        type=float,
                        help='Fraction of X, y used for training.')
    parser.add_argument('-od', '--output_dir', required=False, default=None,
                        help='Optional output directory for saving images')
    parser.add_argument('-rs', '--random_state',
                        required=False,
                        default=42, 
                        type=int)
    args = parser.parse_args()

    X_train = pickle.load(open(args.xt, "rb"))
    y_train = pickle.load(open(args.yt, "rb"))

    if args.xv is not None:
        X_test= pickle.load(open(args.xv, "rb"))
    else:
        X_test = None
    if args.yv is not None:
        y_test= pickle.load(open(args.yv, "rb"))
    else:
        y_test = None
    
    random_state = args.random_state
    train_fr = args.training_fraction
    if train_fr is not None and train_fr> 0. and train_fr < 1.:
        X_train, X_test, y_train, y_test = train_test_split(
                                                X_train, 
                                                y_train, 
                                                train_size=train_fr,
                                                random_state=random_state)
    
    rf = get_forest(X_train, y_train, 
                    random_state=random_state, 
                    n_estimators=2000)
    if X_test is not None:
        analyze_on_test_data(rf, X_test=X_test, y_test=y_test, X_train=X_train)
    


