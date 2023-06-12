import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
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


# TODO add check for whether the prediction is a distribution or a point estimate
# (i.e. whether it is an exact GP or a variational GP)

def plot_gp(func):
    """
    Decorator to plot the GP predictions
    """
    def wrapper(pred_train, pred_test, time_train, time_test, y_train, y_test):
        """ 
        Wrapper function to plot the GP predictions

        Args:
            pred_train (gpytorch.distributions.MultivariateNormal): GP model f(x) for the training data
            pred_test (gpytorch.distributions.MultivariateNormal): GP model f(x) for the test data
            time_train (torch.Tensor): time for the training data
            time_test (torch.Tensor): time for the test data
            y_train (torch.Tensor): PV production for the training data
            y_test (torch.Tensor): PV production for the test data
        
        Returns:
            wrapper (function): function to plot the GP predictions
        """
        plt.figure(figsize=(13, 5))

        plt.scatter(time_train, y_train, marker='x', color='black', alpha=0.7, label='Observed Data')
        plt.scatter(time_test, y_test, marker='x', color='black', alpha=0.7)

        func(pred_train, pred_test, time_train, time_test, y_train, y_test)

        plt.legend(loc='upper left')

        plt.xlabel('Time (5 min intervals between 8am and 4pm)', fontsize=13)
        plt.ylabel('PV Production (0-1 Scale)')

        plt.show()
    
    return wrapper

@plot_gp
def plot_gp_predictions(pred_train, pred_test, time_train, time_test, y_train, y_test):
    lower_train, upper_train = pred_train.confidence_region()
    lower_test, upper_test = pred_test.confidence_region()

    plt.plot(time_train, pred_train.mean, color='b')
    plt.fill_between(time_train, lower_train, upper_train, alpha=0.1, color='b')

    plt.plot(time_test, pred_test.mean, color='r')
    plt.fill_between(time_test, lower_test, upper_test, alpha=0.1, color='r')
    plt.vlines(x=time_train.min() + len(time_train), ymin=0, ymax=max(y_train.max(), y_test.max()), 
               color='black', linestyle='--', label='Train-Test Split')

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