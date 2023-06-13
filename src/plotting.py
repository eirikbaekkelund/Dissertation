import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import torch
import gpytorch
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

def plot_train_test_split(time_train, time_test, y_train, y_test):
    """ 
    Plot the train-test split of the data

    Args:
        time_train (torch.Tensor): training time
        time_test (torch.Tensor): test time
        y_train (torch.Tensor): training data
        y_test (torch.Tensor): test data
    """
    # set figure size
    plt.figure(figsize=(15, 6))

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


def plot_gp(model, x_train, x_test, y_train, y_test, inducing_points = None, y_inducing = None, device=torch.device('cpu')):
    
    """ 
    Plot the GP model predictions

    Args:
        model (gpytorch.models.ApproximateGP): GP model
        x_train (torch.Tensor): training data
        x_test (torch.Tensor): test data
        inducing_points (torch.Tensor): inducing points  
    """
    preds_train = model.predict(x_train, device=device)
    preds_test = model.predict(x_test, device=device)

    plt.figure(figsize=(15, 6))

    with torch.no_grad():

        if isinstance(model.likelihood, gpytorch.likelihoods.GaussianLikelihood):
            # plot the means
            plt.plot(x_train, preds_train.mean, color='b')
            plt.plot(x_test, preds_test.mean, color='r')

            # plot the confidence regions
            lower, upper = preds_train.confidence_region()
            plt.fill_between(x_train, lower, upper, color='b', alpha=0.1)

            lower, upper = preds_test.confidence_region()
            plt.fill_between(x_test, lower, upper, color='r', alpha=0.1)
        
        else:
            # plot the means
            plt.plot(x_train, preds_train.mean.mean(axis=0), color='b')            
            plt.plot(x_test, preds_test.mean.mean(axis=0), color='r')

            # plot the confidence regions
            lower_train, upper_train = np.percentile(preds_train.mean, q=[5, 95], axis=0)
            plt.fill_between(x_train, lower_train, upper_train, alpha=0.1, color='b')
            
            lower_test, upper_test = np.percentile(preds_test.mean, q=[2.5, 97.5], axis=0)
            plt.fill_between(x_test, lower_test, upper_test, alpha=0.1, color='r')
        
        # scatter test data
        plt.scatter(x_test, y_test, color='k', alpha=0.4, marker='x')
        
        if isinstance(model, gpytorch.models.ApproximateGP):
            inducing_points = model.variational_strategy.inducing_points
            # scatter inducing points
            plt.scatter(inducing_points, y_inducing, color='k', marker='x', label='Observed Data', alpha=0.4)
        
        else:
            plt.scatter(x_train, y_train, color='k', marker='x', label='Observed Data', alpha=0.4)
       
    ymax = max(y_train.max(), y_test.max()) + 0.1
    
    plt.vlines(x=x_train.min() + len(x_train), ymin=-0.05, ymax=ymax, 
               color='black', linestyle='--', label='Train-Test Split') 
    
    plt.ylim(-0.01, ymax)
    plt.xlabel('Time (5 min intervals between 8am and 4pm)', fontsize=13)
    plt.ylabel('PV Production (0-1 Scale)', fontsize=13)

    plt.legend(loc='upper left')

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