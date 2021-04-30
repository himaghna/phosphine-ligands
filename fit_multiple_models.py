from argparse import ArgumentParser
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from plotting_functions import plot_parity
import multiple_models


if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument('-xt', help='Path of X.p')
    parser.add_argument('-yt', help='Path of y.p')
    parser.add_argument('-m', '--model',
                        help='Type of ml algorithm. [rf, gpr]')
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
    model_name = args.model
    random_state = args.random_state
    train_fr = args.training_fraction

    if args.xv is not None:
        X_test= pickle.load(open(args.xv, "rb"))
    else:
        X_test = None
    if args.yv is not None:
        y_test= pickle.load(open(args.yv, "rb"))
    else:
        y_test = None
    
    if train_fr is not None and train_fr> 0. and train_fr < 1.:
        X_train, X_test, y_train, y_test = train_test_split(
                                                X_train, 
                                                y_train, 
                                                train_size=train_fr,
                                                random_state=random_state)
    
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    if model_name == 'rf' or model_name == 'random_forest':
        model_ = multiple_models.get_forest(X_train, y_train, 
                                            random_state=random_state, 
                                            n_estimators=2000)
    elif model_name == 'gpr':
        model_ = multiple_models.get_gpr(X_train, y_train, 
                                         random_state=random_state, 
                                         n_restarts_optimizer=10,
                                         normalize_y=True)
    if X_test is not None:
        X_test= x_scaler.transform(X_test)
        y_pred, cli_95 = multiple_models.predict_on_test_data(
                                                        model_, 
                                                        X_test=X_test,
                                                        model_type=model_name,
                                                        X_train=X_train)
        multiple_models.analyze_on_test_data(y_test=y_test, 
                                             y_test_pred=y_pred, 
                                             cli_95=cli_95)
        
    


