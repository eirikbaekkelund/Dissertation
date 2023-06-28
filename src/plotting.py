import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import torch
import gpytorch
from scipy.stats import beta
from mpl_toolkits.basemap import Basemap

def plot_grid(df, COORDS, RADIUS):
    """
    Plot the grid of the UK with the PV systems

    Args:
        df (pd.DataFrame): dataframe containing the PV systems
        COORDS (tuple): coordinates of the center of the circle
        RADIUS (float): radius of the circle
    """
    _, ax = plt.subplots(figsize=(8, 8))

    # create Basemap instance
    map_uk = Basemap(llcrnrlon=-7, llcrnrlat=49, urcrnrlon=2, urcrnrlat=60, resolution='l', ax=ax)

    # draw coastlines and country boundaries
    map_uk.drawcoastlines()
    map_uk.drawcountries()

    # add some nice background colours
    map_uk.fillcontinents(color='forestgreen', lake_color='lightblue', alpha=0.5)

    # Plot PV systems
    x, y = map_uk(df['longitude_noisy'].values, df['latitude_noisy'].values)
    map_uk.scatter(x, y, alpha=0.4, color='b', label='PV systems')

    # a circle representing the desired area
    lon, lat = map_uk(COORDS[1], COORDS[0])
    circle = plt.Circle((lon, lat), RADIUS, color='r', fill=True, alpha=0.3, label='Selected Area')
    ax.add_patch(circle)


    ax.set_xticks(np.arange(-7, 3,))
    ax.set_yticks(np.arange(49, 61))

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('PV systems in the UK')
    ax.legend()

    plt.show()

def plot_train_test_split(y_train, y_test):
    """ 
    Plot the train-test split of the data

    Args:

        y_train (torch.Tensor): training data
        y_test (torch.Tensor): test data
    """
    # set figure size
    plt.figure(figsize=(15, 6))

    time_train = torch.arange(0, len(y_train))
    time_test = torch.arange(len(y_train), len(y_train) + len(y_test))

    # plot the training and test data
    plt.plot(time_train, y_train, color='b', alpha=0.4)
    plt.plot(time_test, y_test, color='r', alpha=0.4)

    # plot the train-test split cutoff
    plt.vlines( x= time_train[0] + len(time_train), 
                ymin=-0.01, 
                ymax=max(y_train.max(), y_test.max()), 
                label='Train-Test Split', 
                color='black', 
                linestyle='--')

    plt.xlabel('Time (5 min intervals between 8am and 4pm)', fontsize=13)
    plt.ylabel('PV Production (0-1 Scale)', fontsize=13)
    
    plt.legend()
    plt.show();


def mode_beta_dist(alpha, beta):
    """ 
    Calculate the mode of a beta distribution given the alpha and beta parameters

    Args:
        alpha (torch.Tensor): alpha parameter
        beta (torch.Tensor): beta parameter
    
    Returns:
        mode (torch.Tensor): mode of the beta distribution
    """
    result = np.zeros_like(alpha)  # Initialize an array of zeros with the same shape as alpha

    mask_alpha_gt_1 = alpha > 1
    mask_beta_gt_1 = beta > 1
    mask_alpha_eq_beta = alpha == beta
    mask_alpha_le_1 = alpha <= 1
    mask_beta_le_1 = beta <= 1

    result[mask_alpha_gt_1 & mask_beta_gt_1] = (alpha[mask_alpha_gt_1 & mask_beta_gt_1] - 1) / (alpha[mask_alpha_gt_1 & mask_beta_gt_1] + beta[mask_alpha_gt_1 & mask_beta_gt_1] - 2)
    result[mask_alpha_eq_beta] = 0.5
    result[mask_alpha_le_1 & mask_beta_gt_1] = 0
    result[mask_alpha_gt_1 & mask_beta_le_1] = 1

    return result

def plot_gp(model, x_train, x_test, y_train, y_test, y_inducing=None, pred_type='mean'):
    """
    Plot the GP predictions for a given model and data

    Args:
        model (gpytorch.models.GP): GP model
        x_train (torch.Tensor): training inputs
        x_test (torch.Tensor): test inputs
        y_train (torch.Tensor): training targets
        y_test (torch.Tensor): test targets
        y_inducing (torch.Tensor): inducing points
        pred_type (str): type of prediction to plot
    """
    assert pred_type in ['mean', 'median', 'both', 'none'], 'pred_type must be one of [mean, median, both, none]'
    assert isinstance(model.likelihood, gpytorch.likelihoods.BetaLikelihood) or isinstance(model.likelihood, gpytorch.likelihoods.GaussianLikelihood), 'Unknown likelihood'
    
    # time points for the training and test data
    time_train = torch.arange(0, len(y_train))
    time_test = torch.arange(len(y_train), len(y_train) + len(y_test))
    
    plt.figure(figsize=(15, 6))
    
    
    def plot_observed_data():
        """ 
        Scatters the observed data
        """
        if y_inducing is not None:
            inducing_points = torch.arange(0, y_inducing.size(0))
            plt.scatter(inducing_points, y_inducing, color='k', marker='x', label='Observed Data', alpha=0.4)
        else:
            plt.scatter(time_train, y_train, color='k', marker='x', label='Observed Data', alpha=0.4)
    
    def plot_beta_mode_predictions():
        """ 
        Plots the mean and confidence intervals of the beta distribution
        from MC samples using the mode of the distribution
        """
        model.predict(x_train, device=torch.device('cpu'))
        alphas_train = model.likelihood.alpha
        betas_train = model.likelihood.beta
        
        modes_train = mode_beta_dist(alphas_train, betas_train)
        mode_mean_train = np.mean(modes_train, axis=0)
        mode_percentile_train = np.percentile(modes_train, q=[2.5, 97.5], axis=0)

        plt.plot(time_train, mode_mean_train, color='g', label='Mode')
        plt.fill_between(time_train, mode_percentile_train[0], mode_percentile_train[1], color='g', alpha=0.1, label='95% Confidence Interval (Mode)')

        model.predict(x_test, device=torch.device('cpu'))
        alphas_test = model.likelihood.alpha
        betas_test = model.likelihood.beta

        modes_test = mode_beta_dist(alphas_test, betas_test)
        mode_mean_test = np.mean(modes_test, axis=0)
        mode_percentile_test = np.percentile(modes_test, q=[2.5, 97.5], axis=0)

        plt.plot(time_test, mode_mean_test, color='g')
        plt.fill_between(time_test, mode_percentile_test[0], mode_percentile_test[1], color='g', alpha=0.1)
    
    def plot_gaussian_predictions():
        """ 
        Plots the mean and confidence intervals of the gaussian distribution
        """
        preds_train = model.predict(x_train, device=torch.device('cpu'))
        preds_test = model.predict(x_test, device=torch.device('cpu'))

        with torch.no_grad():
            # plot the means
            plt.plot(time_train, preds_train.mean, color='b')
            plt.plot(time_test, preds_test.mean, color='r')

            # plot the confidence regions
            lower, upper = preds_train.confidence_region()
            plt.fill_between(time_train, lower, upper, color='b', alpha=0.1)

            lower, upper = preds_test.confidence_region()
            plt.fill_between(time_test, lower, upper, color='r', alpha=0.1)
    
    def plot_multivariate_predictions():
        """ 
        Plots the mean and confidence intervals of the mean and
        median of the posterior distribution when using non-Gaussian likelihoods
        """
        dist_train = model.predict(x_train, device=torch.device('cpu'))
        dist_test = model.predict(x_test, device=torch.device('cpu'))
        preds_train = dist_train.sample((50,))
        preds_test = dist_test.sample((50,))

        def plot_mean():
            """
            Plot the mean and intervals from MC samples of the mean
            """
            mean_preds_train = preds_train.mean(axis=0)
            mean_preds_test = preds_test.mean(axis=0)
            
            # plot the means
            plt.plot(time_train, mean_preds_train.mean(axis=0), color='b', label='Mean')           
            plt.plot(time_test, mean_preds_test.mean(axis=0), color='b')
            
            # plot the confidence regions
            lower_train, upper_train = np.percentile(mean_preds_train, q=[2.5, 97.5], axis=0)
            plt.fill_between(time_train, lower_train, upper_train, alpha=0.1, color='b', label='95% Confidence Interval (Mean)')

            lower_test, upper_test = np.percentile(mean_preds_test, q=[2.5, 97.5], axis=0)
            plt.fill_between(time_test, lower_test, upper_test, alpha=0.1, color='b')

        def plot_median():
            """ 
            Plot the mean and intervals from MC samples of the median
            """
            median_preds_train = preds_train.median(axis=0).values
            median_preds_test = preds_test.median(axis=0).values

            plt.plot(time_train, median_preds_train.mean(axis=0), color='r', label='Median')
            plt.plot(time_test, median_preds_test.mean(axis=0), color='r')

            lower_train, upper_train = np.percentile(median_preds_train, q=[2.5, 97.5], axis=0)
            plt.fill_between(time_train, lower_train, upper_train, alpha=0.1, color='r', label='95% Confidence Interval (Median)')

            lower_test, upper_test = np.percentile(median_preds_test, q=[2.5, 97.5], axis=0)
            plt.fill_between(time_test, lower_test, upper_test, alpha=0.1, color='r')
        
        if pred_type == 'mean':
            plot_mean()
        elif pred_type == 'median':
            plot_median()
        
        elif pred_type == 'both':
            plot_mean()
            plot_median()
        elif pred_type == 'none':
            pass
    
    # scatter observed data
    plot_observed_data()

    if isinstance(model.likelihood, gpytorch.likelihoods.BetaLikelihood):
        plot_beta_mode_predictions()
        plot_multivariate_predictions()
    
    elif isinstance(model.likelihood, gpytorch.likelihoods.GaussianLikelihood):
        plot_gaussian_predictions()
    
    else:
        print('Unknown likelihood')
    
    # scatter test data
    plt.scatter(time_test, y_test, color='k', alpha=0.4, marker='x')

    ymax = max(y_train.max(), y_test.max()) + 0.1

    plt.vlines(x=time_train.max(), ymin=-0.05, ymax=ymax, color='black', linestyle='--', label='Train-Test Split')

    plt.ylim(-0.01, ymax)
    plt.xlabel('Time (5 min intervals between 8am and 4pm)', fontsize=13)
    plt.ylabel('PV Production (0-1 Scale)', fontsize=13)

    plt.legend(loc='upper left')
    plt.show();

def plot_alpha_beta(model):
    fig, ax = plt.subplots(figsize=(15, 6), sharey=False)
    time = torch.arange(0, len(model.likelihood.alpha.mean(axis=0)))

    ax.scatter(time, model.likelihood.beta.mean(axis=0), label='Beta', color='b', alpha=0.2)
    ax2 = ax.twinx()
    ax2.scatter(time, model.likelihood.alpha.mean(axis=0), label='Alpha', color='r', alpha=0.2)
    ax.set_ylabel('Beta')
    ax2.set_ylabel('Alpha')
    ax.set_xlabel('Time')

    # show legend for both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.show()

def plot_acf_pacf(y):
    """
    Plot the acf and pacf for all systems in a grid

    Args:
        y (torch.tensor): PV production
    """
    _, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    for i in range(y.shape[1]):
        plot_acf(y[:, i], ax=ax[0], alpha=0.2, lags=len(y) // 2, title='ACF', color='b')
        plot_pacf(y[:, i], ax=ax[1], alpha=0.2, lags=len(y) // 2 - 1, title='PACF', color='b', method='ywm')
    
    plt.xlabel('Lag')
    plt.show()