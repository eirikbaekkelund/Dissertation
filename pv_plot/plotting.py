import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import gpytorch
from mpl_toolkits.basemap import Basemap
from typing import Optional

def plot_grid(df, coords, radius=1, distance_method='circle'):
    """
    Plot the grid of the UK with the PV systems

    Args:
        df (pd.DataFrame): dataframe containing the PV systems locations
        COORDS (tuple): coordinates of the center of the circle / corners of the polygon
        RADIUS (float): radius of the circle
    """
    assert distance_method in ['circle', 'poly'], 'distance_method must be either "circle" or "poly"'
    if distance_method == 'circle':
        assert len(coords) == 2, 'coords must be a tuple of length 2'
    
    elif distance_method == 'poly':
        assert len(coords) == 4, 'coords must be a tuple of length 4'
    
    plt.rcParams['font.family'] = 'Arial'
    _, ax = plt.subplots(figsize=(8, 8))

    # create Basemap instance
    map_uk = Basemap(llcrnrlon=-7, llcrnrlat=49, urcrnrlon=2, urcrnrlat=60, resolution='l', ax=ax)

    # draw coastlines and country boundaries
    map_uk.drawcoastlines()
    map_uk.drawcountries()

    # add some nice background colours
    map_uk.fillcontinents(color='forestgreen', lake_color='lightblue', alpha=0.5)

    # Plot PV systems
    if 'latitude_noisy' in df.columns:
        x, y = map_uk(df['longitude_noisy'].values, df['latitude_noisy'].values)
    
    elif 'latitude' in df.columns:
        x, y = map_uk(df['longitude'].values, df['latitude'].values)

    map_uk.scatter(x, y, alpha=0.4, color='b', label='PV systems')

    if distance_method == 'circle':
        # a circle representing the desired area
        lon, lat = map_uk(coords[1], coords[0])
        circle = plt.Circle((lon, lat), radius, color='r', fill=True, alpha=0.3, label='Selected Area')
        ax.add_patch(circle)
    
    elif distance_method == 'poly':
        c1, c2, c3, c4 = coords
        lon1, lat1 = map_uk(c1[1], c1[0])
        lon2, lat2 = map_uk(c2[1], c2[0])
        lon3, lat3 = map_uk(c3[1], c3[0])
        lon4, lat4 = map_uk(c4[1], c4[0])
        poly = plt.Polygon([(lon1, lat1), (lon2, lat2), (lon4, lat4), (lon3, lat3)], color='r', fill=True, alpha=0.3, label='Selected Area')
        ax.add_patch(poly)



    ax.set_xticks(np.arange(-7, 3,))
    ax.set_yticks(np.arange(49, 61))

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('PV systems in the UK')
    ax.legend()

    plt.show()

def scatter_pv_vals(df, day = '2018-01-31 12:00:00'):
    """
    Plot the grid of the UK with the PV systems

    Args:
        df (pd.DataFrame): dataframe containing the PV systems locations
        COORDS (tuple): coordinates of the center of the circle / corners of the polygon
        RADIUS (float): radius of the circle
    """

    
    plt.rcParams['font.family'] = 'Arial'
    _, ax = plt.subplots(figsize=(7, 7))

    map_uk = Basemap(llcrnrlon=-7, llcrnrlat=49, urcrnrlon=2, urcrnrlat=60, resolution='l', ax=ax)

    map_uk.drawcoastlines()
    map_uk.drawcountries()
    map_uk.fillcontinents(color='forestgreen', lake_color='lightblue', alpha=0.5)

    # Plot PV systems
    # get first instance where the datetime column is at 12:00
    df = df[df['datetime'] == day]

    if 'latitude_noisy' in df.columns:
        x, y = map_uk(df['longitude_noisy'].values, df['latitude_noisy'].values)
    
    elif 'latitude' in df.columns:
        x, y = map_uk(df['longitude'].values, df['latitude'].values)

    # scatter all the PV systems, but set a color bar based on the current PV value
    pvals = map_uk.scatter(x, y, c=df['PV'].values, cmap='Reds', alpha=0.7, s=20)
    plt.colorbar(pvals, label='PV value')

    ax.set_xticks(np.arange(-7, 3,))
    ax.set_yticks(np.arange(49.5, 58))

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'PV systems in the UK ({day})')

    plt.show()
def plot_train_test_split(y_train, y_test, minute_interval=5):
    """ 
    Plot the train-test split of the data

    Args:

        y_train (torch.Tensor): training data
        y_test (torch.Tensor): test data
    """
    # set figure size
    plt.rcParams['font.family'] = 'Arial'

    plt.figure(figsize=(15, 6))

    time_train = torch.arange(0, len(y_train))
    time_test = torch.arange(len(y_train), len(y_train) + len(y_test))

    # plot the training and test data
    plt.plot(time_train, y_train, color='b', alpha=0.4)
    plt.plot(time_test, y_test, color='r', alpha=0.4)

    # plot the train-test split cutoff
    plt.vlines( x= time_train[0] + len(time_train), 
                ymin=-0.001, 
                ymax=1.001, 
                label='Train-Test Split', 
                color='black', 
                linestyle='--')

    plt.xlabel(f'Time ({minute_interval} min intervals between 8am and 4pm)', fontsize=13)
    plt.ylabel('PV Production (0-1 Scale)', fontsize=13)

    plt.ylim(-0.01, 1.001)
    plt.legend()
    plt.show();

def plot_seasons(train_loader_sp : torch.utils.data.DataLoader,
                 test_loader_sp : torch.utils.data.DataLoader,
                 train_loader_su : torch.utils.data.DataLoader,
                 test_loader_su : torch.utils.data.DataLoader,
                 train_loader_f : torch.utils.data.DataLoader,
                 test_loader_f : torch.utils.data.DataLoader,
                 train_loader_w : torch.utils.data.DataLoader,
                 test_loader_w : torch.utils.data.DataLoader):
    """
    Plot the PV output for each season in a 2x2 grid.

    Args:
        train_loader_sp (torch.utils.data.DataLoader): train loader for spring
        test_loader_sp (torch.utils.data.DataLoader): test loader for spring
        train_loader_su (torch.utils.data.DataLoader): train loader for summer
        test_loader_su (torch.utils.data.DataLoader): test loader for summer
        train_loader_f (torch.utils.data.DataLoader): train loader for fall
        test_loader_f (torch.utils.data.DataLoader): test loader for fall
        train_loader_w (torch.utils.data.DataLoader): train loader for winter
        test_loader_w (torch.utils.data.DataLoader): test loader for winter
    """

    plt.rcParams['font.family'] = 'Arial'

    fig, ax = plt.subplots(2, 2, figsize=(40, 20), sharex=True, sharey=True)
    
    ax[0][0].set_ylabel('PV Output (0-1 Scale)', fontsize=20)
    ax[1][0].set_ylabel('PV Output (0-1 Scale)', fontsize=20)
    
    ax = ax.flatten()
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']

    alpha = 0.1

    for i, (train_loader, test_loader) in enumerate([(train_loader_sp, test_loader_sp), 
                                    (train_loader_su, test_loader_su),
                                    (train_loader_f, test_loader_f),
                                    (train_loader_w, test_loader_w)]):
        season = seasons[i]
        for (x_tr, y_tr), (x_te, y_te) in zip(train_loader, test_loader):
            for j in range(y_tr.shape[1]):
                # time indices for train and test are seperate for each sample
                t_train = torch.arange(y_tr.shape[0])
                t_test = torch.arange(y_tr.shape[0], y_te.shape[0] + y_tr.shape[0])
                
                ax[i].plot(t_train, y_tr[:,j], color='blue', alpha=alpha)
                ax[i].plot(t_test, y_te[:,j], color='red', alpha=alpha)

        ax[i].set_title(season, fontsize=20)
    
    plt.tight_layout()
    plt.show()


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


def plot_gp_ax(model, x_train, x_test, y_train, y_test, pred_type='mode', ax=None):
    """
    Plot the GP predictions for a given model and data

    Args:
        model (gpytorch.models.GP): GP model
        x_train (torch.Tensor): training inputs
        x_test (torch.Tensor): test inputs
        y_train (torch.Tensor): training targets
        y_test (torch.Tensor): test targets
        pred_type (str): type of prediction to plot
        title (str): title for the plot
        ax (matplotlib.axes.Axes): subplot axes to plot on (optional)
    """
    assert pred_type in ['mean', 'median', 'mode', 'all'], 'pred_type must be one of [mean, median, all, none]'
    assert isinstance(model.likelihood, gpytorch.likelihoods.BetaLikelihood) or isinstance(model.likelihood, gpytorch.likelihoods.GaussianLikelihood), 'Unknown likelihood'
    
    # time points for the training and test data
    time_train = torch.arange(0, len(y_train))
    time_test = torch.arange(len(y_train), len(y_train) + len(y_test))
 
    
    dist_train = model.predict(x_train)
    dist_test = model.predict(x_test)
    preds_train = dist_train.sample((100,))
    preds_test = dist_test.sample((100,))
    
    lower_train, upper_train = np.percentile(preds_train, q=[2.5, 97.5], axis=0)
    lower_train, upper_train = lower_train.mean(axis=0), upper_train.mean(axis=0)

    lower_test, upper_test = np.percentile(preds_test, q=[2.5, 97.5], axis=0)
    lower_test, upper_test = lower_test.mean(axis=0), upper_test.mean(axis=0)

    ax.fill_between(time_train, lower_train, upper_train, alpha=0.1, color='b')
    ax.fill_between(time_test, lower_test, upper_test, alpha=0.1, color='r')


    def plot_mean():
        """Plot the mean and intervals from MC samples of the mean"""
        mean_preds_train = preds_train.mean(axis=0)
        mean_preds_test = preds_test.mean(axis=0)
        
        # plot the means
        ax.plot(time_train, mean_preds_train.mean(axis=0), color='y', label='Mean')           
        ax.plot(time_test, mean_preds_test.mean(axis=0), color='y')
    
    def plot_median():
        """Plot the mean and intervals from MC samples of the median"""
        median_preds_train = preds_train.median(axis=0).values
        median_preds_test = preds_test.median(axis=0).values

        ax.plot(time_train, median_preds_train.mean(axis=0), color='b', label='Median')
        ax.plot(time_test, median_preds_test.mean(axis=0), color='r')

    
    def plot_mode():
        """Plots the mean and confidence intervals of the beta distribution from MC samples using the mode of the distribution"""
        model.predict(x_train)
        alphas_train = model.likelihood.alpha
        betas_train = model.likelihood.beta
        
        modes_train = mode_beta_dist(alphas_train, betas_train)
        mode_mean_train = np.mean(modes_train, axis=0)

        ax.plot(time_train, mode_mean_train, color='g', label='Mode')

        model.predict(x_test)
        alphas_test = model.likelihood.alpha
        betas_test = model.likelihood.beta

        modes_test = mode_beta_dist(alphas_test, betas_test)
        mode_mean_test = np.mean(modes_test, axis=0)

        ax.plot(time_test, mode_mean_test, color='g')
    
    if pred_type == 'mean':
        plot_mean()
    elif pred_type == 'median':
        plot_median()
    elif pred_type == 'mode':
        plot_mode()
    elif pred_type == 'all':
        plot_mean()
        plot_median()
        plot_mode()
    
    ax.scatter(time_train, y_train, color='k', marker='x', label='Observed Data', alpha=0.5)
    ax.scatter(time_test, y_test, color='k', alpha=0.5, marker='x')
    ax.vlines(x=time_train.max(), ymin=-0.05, ymax=1.001, color='black', linestyle='--', label='Train-Test Split')
    ax.set_ylim(-0.01, 1.01)

    
def plot_alpha_beta(model):
    
    fig, ax = plt.subplots(figsize=(15, 6), sharey=False)
    plt.rcParams['font.family'] = 'Arial'
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

def boxplot_forecast_horizon(df_dict, pred_points=24, season : Optional[str] = None):
    
    fig, ax = plt.subplots(len(df_dict) //2 , 2, figsize=(30, 20), sharey=True, sharex=True)
    plt.rcParams.update({'font.family': 'Arial'})
    ax = ax.flatten()
    
    for i,  (key, df) in enumerate(df_dict.items()):
        df_transposed = df.T
        # boxplot the data with no fill but with a line at the median
        sns.boxplot(data=df_transposed, ax=ax[i], showfliers=False, medianprops={'color':'red'}, showmeans=True, color='white')
        ax[i].set_title(key, fontsize=20)
        # set y axis label on leftmost plots
        if i % 2 == 0:
            ax[i].set_ylabel('MAE Errors', fontsize=15)
            # set ticksizes
            ax[i].tick_params(axis='both', which='major', labelsize=15)
        # set x axis label on bottom plots
        if i >= len(df_dict) - 2:
            ax[i].set_xlabel('Time Step', fontsize=20)
            # set ticks to be from 1 to number of time steps
            # and let 1 be where the 0 is
            ax[i].set_xticks(np.arange(0, pred_points), np.arange(1, pred_points + 1))
            ax[i].tick_params(axis='both', which='major', labelsize=15)

    if season is not None:
        title = season[0].upper() + season[1:]
        fig.suptitle(title, fontsize=25)

    plt.tight_layout()
    plt.show();

def boxplot_models(results : dict, season : Optional[str] = None):
    # boxplot of models MAE (mean and median)
    
    plt.figure(figsize=(15, 7))
    plt.rcParams.update({'font.family': 'Arial'})
    plt.boxplot([results[key]['mean'] for key in results.keys()], labels=results.keys(), showmeans=True)
    plt.ylabel('Mean MAE')

    if season is not None:
        title = season[0].upper() + season[1:]
        plt.title(title)
    plt.tight_layout()
    plt.show();

def plot_forecast_mae(results : dict, season : Optional[str] = None, pred_points=24):
    colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    

    plt.figure(figsize=(15, 7))
    plt.rcParams.update({'font.family': 'Arial'})
    
    for i, key in enumerate(results.keys()):
        mean = results[key]['mean']
        plt.plot(np.arange(0, pred_points), mean, color=colors[i], linestyle=linestyles[i], label=key)
    
    plt.xlabel('Forecasting Steps Ahead (5 min Intervals)')
    plt.ylabel('Mean MAE')
    plt.xticks(np.arange(0, pred_points), np.arange(1, pred_points + 1))

    if season is not None:
        title = season[0].upper() + season[1:]
        plt.title(title)
    
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_lfm(lfm, dataset):
    """
    Plot the latent force and the PV predictions

    Args:
        lfm (models.LFM): LFM model
        dataset (datasets.PVData): dataset
    """

    pred_cloud = lfm(dataset.times)
    cloud_mean = pred_cloud.mean.detach().squeeze(0).numpy()
    cloud_var = pred_cloud.variance.detach().squeeze(0).numpy()
    cloud_std = np.sqrt(cloud_var)

    pv_init = dataset.pv_init
    pv_var = cloud_var[0] # assume casual relationship between cloud and PV
    pv_dist = lfm.predict_pv(pv_init, pv_var, dataset.times)

    pv_mean = pv_dist.mean.detach().squeeze(0).numpy()
    pv_var = pv_dist.variance.detach().squeeze(0).numpy()
    pv_std = np.sqrt(pv_var)

    cloud_init_test = cloud_mean[-1]
    cloud_var_test = cloud_var[-1] 
    cloud_dist_test = lfm.predict_cloud(cloud_init_test, cloud_var_test, dataset.times_test)

    cloud_mean_test = cloud_dist_test.mean.detach().squeeze(0).numpy()
    cloud_var_test = cloud_dist_test.variance.detach().squeeze(0).numpy()
    cloud_std_test = np.sqrt(cloud_var_test)

    pv_init_test = pv_mean[-1]
    pv_var_test = cloud_var_test[0]
    pv_dist_test = lfm.predict_pv(pv_init_test, pv_var_test, dataset.times_test)

    pv_mean_test = pv_dist_test.mean.detach().squeeze(0).numpy()
    pv_var_test = pv_dist_test.variance.detach().squeeze(0).numpy()
    pv_std_test = np.sqrt(pv_var_test)

    plt.rcParams['font.family'] = 'Arial'

    fig, ax = plt.subplots(1, 1, figsize=(20, 8))

    ax.scatter(dataset.times, dataset.cloud, marker='o', alpha=0.5, color='k', label='Cloud Data')
    ax.scatter(dataset.times, dataset.pv, marker='x', alpha=0.5, color='k', label='PV Data')
    ax.scatter(dataset.times_test, dataset.cloud_test, marker='o', alpha=0.5, color='k')
    ax.scatter(dataset.times_test, dataset.pv_test, marker='x', alpha=0.5, color='k')

    ax.plot(dataset.times, cloud_mean, label='Latent Force', color='b')
    ax.fill_between(dataset.times, cloud_mean - 1.96 * cloud_std, cloud_mean + 1.96 * cloud_std, alpha=0.1, color='b')

    ax.plot(dataset.times, pv_mean, label='Predicted', color='r')
    ax.fill_between(dataset.times, pv_mean - 1.96 * pv_std, pv_mean + 1.96 * pv_std, alpha=0.1, color='r')

    ax.plot(dataset.times_test, cloud_mean_test, color='b')
    ax.fill_between(dataset.times_test, cloud_mean_test - 1.96 * cloud_std_test, cloud_mean_test + 1.96 * cloud_std_test, alpha=0.1, color='b')

    ax.plot(dataset.times_test, pv_mean_test, color='r')
    ax.fill_between(dataset.times_test, pv_mean_test - 1.96 * pv_std_test, pv_mean_test + 1.96 * pv_std_test, alpha=0.1, color='r')

    y_min = min(min(cloud_mean - 1.96 * cloud_std), min(pv_mean - 1.96 * pv_std), 
                min(cloud_mean_test - 1.96 * cloud_std_test), min(pv_mean_test - 1.96 * pv_std_test))
    y_max = max(max(cloud_mean + 1.96 * cloud_std), max(pv_mean + 1.96 * pv_std),
                max(cloud_mean_test + 1.96 * cloud_std_test), max(pv_mean_test + 1.96 * pv_std_test))

    ylim = (y_min - 0.05, y_max + 0.05)
    ax.legend(loc='upper left', fontsize=20)
    ax.set_xlabel('Time Steps', fontsize=20)
    ax.set_ylabel('PV / Cloud Cover', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylim(y_min -0.05, y_max + 0.05)
    ax.axvspan(dataset.times[0], dataset.times[-1], alpha=0.2, color='gray')
    arrow = ax.arrow(dataset.times[-1], y_max - 0.05, 10, 0, head_width=0.02, head_length=0.5, fc='k', ec='k')
    # mark initial point of test data with a blue and red circle with no fill
    ax.scatter(dataset.times[-1], cloud_init_test, marker='o', color='b', edgecolors='k', s=100)
    ax.scatter(dataset.times[-1], pv_init_test, marker='o', color='r', edgecolors='k', s=100)
    # Add text above the arrow
    ax.annotate('Extrapolation', xy=(dataset.times[-1] + 5, y_max - 0.02), fontsize=20, ha='center')

    plt.tight_layout()
    plt.show()
