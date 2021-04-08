from os import mkdir
from os.path import isdir, join

import matplotlib
import matplotlib.pyplot as plt
import torch

from tensor_ops import sort_tensor, detach_tensor_to_numpy


def plot_maes(maes, out_dir, response_name, 
              mae_scale=None, mae_scale_label='MAE scale', 
              interactive_run=True):
    """Plot the Mean Absolute Errors

    Parameters
    ----------
    maes: List(float)
        Sequence of maes.
    out_dir: str
        Path of directory to output plot.
    response_name: str
        Label of the response.
    mae_scale: float
        All maes are scaled with resepect to this.
    mae_scale_label: str
        Label for the mae scaling factor.

    """
    plt.rcParams['svg.fonttype'] = 'none'
    x_label = 'Optimization Step'
    y_label = f'{response_name} Test MAE'
    legend_prefix = 'Minimum MAE'
    if mae_scale:
        maes = [mae / mae_scale for mae in maes]
        y_label += f' Scaled by {mae_scale_label}' 
        legend_prefix += f' / {mae_scale_label}'

    plt.plot([_ for _ in range(len(maes))], maes,
             marker='s', markerfacecolor='m', markeredgecolor='black', 
             c='m', markersize=0.1,
             markeredgewidth=0.01)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.text(0.2, 0.9,
             legend_prefix + ': {:.2f}'.format(min(maes)), 
             transform=plt.gca().transAxes,
             fontsize=16)
    if interactive_run:
        plt.show()
    else:
        if not isdir(out_dir):
            mkdir(out_dir)
        out_fpath = join(out_dir, 'mae-plot.svg')
        print(f'Saving to {out_fpath}')
        plt.savefig(out_fpath)
        plt.clf()

def plot_y_test_means(y_test_means, out_dir, response_name, interactive_run=True):
    """Plot the Mean value of responses in the test set

    Parameters
    ----------
    y_test_means: List(float)
        Sequence of mean of y_test values.
    out_dir: str
        Path of directory to output plot.
    response_name: str
        Label of the response.

    """
    plt.rcParams['svg.fonttype'] = 'none'
    x_label = 'Optimization Step'
    y_label = f'Mean {response_name} in Test Set'

    plt.plot([_ for _ in range(len(y_test_means))], y_test_means,
             marker='s', markerfacecolor='m', markeredgecolor='black', 
             c='m', markersize=0.1,
             markeredgewidth=0.01)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if interactive_run:
        plt.show()
    else:
        if not isdir(out_dir):
            mkdir(out_dir)
        out_fpath = join(out_dir, 'y_test_means-plot.svg')
        print(f'Saving to {out_fpath}')
        plt.savefig(out_fpath)
        plt.clf()
    
def plot_acq_func(acq_func, X_test, X_train, xlabel, 
                  X_new=None, out_dir=None, out_fname=None,
                  interactive_run=True):
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
    if interactive_run:
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

def plot_testing(model, 
                 X_test, X_train, 
                 y_train, 
                 xlabel,
                 ylabel,
                 visualization_dim,
                 interactive_run=True,
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
    X_test, idx = sort_tensor(X_test, sort_column_id=visualization_dim)
    y_test = y_test.index_select(0, idx)
    with torch.no_grad():
        posterior = model.posterior(X_test)
        posterior_mean = posterior.mean.cpu().numpy()
        lower, upper = posterior.mvn.confidence_region()
        X_test = X_test[:, visualization_dim]
        ax.plot(X_test.cpu().numpy(), y_test.cpu().numpy(), 
                'lightcoral', label=f'True {ylabel}')
        ax.plot(X_test.cpu().numpy(), posterior_mean,
                'b', label='Posterior Mean')
        ax.fill_between(X_test.cpu().numpy().squeeze(), 
                        lower.cpu().numpy(), upper.cpu().numpy(), 
                        alpha=0.5, label='95% Credibility')   
        ax.scatter(X_train[:, visualization_dim].cpu().numpy(), 
                   y_train.cpu().numpy(),
                   s=150, c='m', marker='*', label='Training Data')

        if X_new is not None:    
            ax.scatter(X_new[:, visualization_dim].cpu().numpy(), 
                       y_new.cpu().numpy(),
                       s=150, c='r', marker='*', label='Infill Data')
        
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    if mean_absolute_error is not None:
        plt.text(0.7,  0.9, 'MAE: {:.2f}'.format(mean_absolute_error), 
                transform=ax.transAxes, fontsize=16)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    if interactive_run:
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


def plot_confidence_region(means, lowers, uppers, 
                           x_label, y_label, x_tick_labels=None,
                           mean_line_color='k', 
                           confidence_region_color='b', 
                           confidence_region_transparency=0.5,
                           legend=None):
    plt.rcParams['svg.fonttype'] = 'none'
    x_vals = [id for id, _ in enumerate(means)]
    means = detach_tensor_to_numpy(means)
    lowers = detach_tensor_to_numpy(lowers)
    uppers = detach_tensor_to_numpy(uppers)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel(y_label, fontsize=20)
    ax.set_xlabel(x_label, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    if x_tick_labels is not None:
        ax.set_xticklabels(x_label, fontsize=15)
    ax.plot(x_vals, means, color=mean_line_color)
    ax.fill_between(x_vals, lowers, uppers, 
                    alpha=confidence_region_transparency, 
                    color=confidence_region_color) 
    if legend is not None:
        ax.legend([legend], fontsize=20)
    plt.show()
