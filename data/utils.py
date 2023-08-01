import numpy as np
import torch
import pandas as pd
import os
import xarray as xr
import datetime
import time
from shapely.geometry import Polygon, Point

def load_data(folder_name, file_name):
    """ 
    Load data file from folder 

    Args:
        folder_name (str): folder
        file_name (str): file name
    
    Returns:
        data (xarray): data
    """    
    
    # get path to the remote data
    # get absolute path to the folder
    
    remote_path = os.path.dirname(os.getcwd())
    # make sure path contains 'Code/'
    assert 'Code' in remote_path, 'Path does not contain Code/ folder'
    
    # if anything follows 'Code/' in the path, remove it
    try:
        if remote_path.split('Code/')[1] != '':
            # remove everything after 'Code/'
            remote_path = remote_path.split('Code/')[0] + f'Code/{folder_name}'
        
    # check that file is in the remote path
    except IndexError:
        remote_path += f'/{folder_name}'

    assert file_name in os.listdir(remote_path)
    
    file_path = os.path.join(remote_path, file_name)
    
    print('==> Loading data')
    
    tic = time.time()
    
    if file_name.endswith('.csv'):
        df = pd.read_csv(file_path)    
    
    elif  file_name.endswith('.netcdf'):
        df = xr.open_dataset(file_path, engine='h5netcdf').to_dataframe()
        df = df.dropna(axis='columns', how='all')
        df = df.clip(lower=0, upper=5e7)
    
    tac = time.time()
    print(f'==> Loaded data in: {int((tac - tic) // 60)} m : {int((tac - tic) % 60)} sec\n')
    
    return df

def set_index(df_location):
    """ 
    Set index of location data to ss_id

    Args:
        df_location (xarray): location data
    
    Returns:
        df_location (xarray): location data with ss_id as index
    """
    assert 'ss_id' in df_location.columns, 'ss_id not in df_location'

    df_location = df_location.set_index('ss_id')
    df_location.index = df_location.index.astype(str)
    
    return df_location

def align_pv_systems(df_location, df_pv):
    """ 
    Align PV systems to locations from metadata and the pv data

    Args:
        df_location (xarray): location data
        df_pv (xarray): PV data
    
    Returns:
        df_pv (xarray): aligned PV data
    """
    assert df_location.index.name == 'ss_id', 'ss_id is not the index of df_location'
    # make sure column data of df_pv and df_location are the same dtype
    df_pv.columns = df_pv.columns.astype(str)
    df_location.index = df_location.index.astype(str)

    pv_system_ids = df_location.index.intersection(df_pv.columns)
    pv_system_ids = np.sort(pv_system_ids)

    df_pv = df_pv[pv_system_ids]
    df_location = df_location.loc[pv_system_ids]
    
    return df_location, df_pv

def scale_by_capacity(df_pv, df_location):
    """ 
    Scale PV data by capacity

    Args:
        df_pv (xarray): PV data
        df_location (xarray): location data
    
    Returns:
        df_pv (xarray): scaled PV data
    """
    capacities = df_location[df_location.index.isin(df_pv.columns)]['kwp'] * 1000
    df_pv = df_pv / capacities
    df_pv = df_pv / df_pv.max()
    
    return df_pv

def drop_night_production(df_pv, threshold=0.4):
    """
    Drop systems which are producing over night

    Args:
        df_pv (pd.DataFrame): PV data
        threshold (float): threshold for night production
    
    Returns:
        df_pv (pd.DataFrame): PV data without systems producing over night
    """
    night_hours = list(range(21, 24)) + list(range(0, 4))
    bad_systems = np.where((df_pv[df_pv.index.hour.isin(night_hours)] > threshold).sum())[0]
    bad_systems = df_pv.columns[bad_systems]

    if len(bad_systems) == 0:
        print('No systems producing over night')
        return df_pv
    
    print(f'Dropping {len(bad_systems)} systems producing over night')
    df_pv.drop(bad_systems, axis='columns', inplace=True)

    return df_pv

def daily_production(df_pv, day_min, day_max):
    """
    Get daily production of PV systems

    Args:
        df_pv (pd.DataFrame): PV data
        day_min (int): last hour of the night
        day_max (int): first hour of the night
    
    Returns:
        df_pv (pd.DataFrame): daily production data from day_min to day_max
    """
    assert day_min < day_max, 'day_min must be smaller than day_max'
    assert day_min >= 0 and day_max <= 24, 'day_min and day_max must be between 0 and 24'

    start_night = datetime.time(day_max, 0)
    end_night = datetime.time(day_min, 0)

    day_index = [time_period for time_period in df_pv.index if time_period.time() < start_night and time_period.time() >= end_night]

    df_pv = df_pv.loc[day_index]

    return df_pv

def remove_zero_production(df_pv, thresh):
    """ 
    Remove systems with zero production for more than thresh percent of the time

    Args:
        df_pv (pd.DataFrame): PV data
        thresh (float): threshold for nan/zero production
    
    Returns:
        df_pv (pd.DataFrame): PV data without systems not producing 
                              for more than thresh percent of the time
    """
    assert (thresh >= 0) and (thresh <= 1), 'thresh must be between 0 and 1'
    # first drop all columns with nan values by nan threshold
    df_pv = df_pv.dropna(axis=1, thresh=len(df_pv) * (1 - thresh) )

    # remaining systems with zero production for more than thresh percent of the time
    selected_cols = df_pv.columns[df_pv.isin([0]).mean() < thresh]
    df_pv = df_pv[selected_cols]

    return df_pv

def get_location_maps(df_location, n_systems):
    """ 
    Create maps for the location data

    Args:
        df_location (pd.DataFrame): location data
        n_systems (int): number of systems to 

    Returns:
        lats_map (dict): latitude map
        long_map (dict): longitude map
    """
    lats = dict(df_location.iloc[:n_systems]['latitude_noisy'])
    longs = dict(df_location.iloc[:n_systems]['longitude_noisy'])

    return lats, longs

def periodic_mapping(time_steps, day_min, day_max, minute_interval):
    """ 
    Create a periodic mapping of time steps to a sine function to 
    capture the periodicity of the data.

    Args:
        time_steps (torch.Tensor): time steps to map
        day_min (int): minimum time of the day
        day_max (int): maximum time of the day
        minute_interval (int): interval between time steps in minutes

    Returns:
        torch.Tensor: mapped time steps to sine function
    """
    total_minutes = (day_max - day_min) * 60  # Total number of minutes in the specified time range
    normalized_minutes = (time_steps * minute_interval) % total_minutes  # Normalize time steps to minutes

    # Apply periodic mapping using sine function
    mapped_values = torch.sin(np.pi * normalized_minutes / total_minutes)
    
    return mapped_values

def save_csv(df, folder_name, file_name):
    """ 
    Save data as csv file

    Args:
        df (pd.DataFrame): data
        folder_name (str): folder
        file_name (str): file name
    """
    
    remote_path = os.path.join(os.path.dirname(os.getcwd()), folder_name)
    # make sure path contains 'Code/' and drop everything after 'Code/'
    remote_path = remote_path.split('Code/')[0] + f'Code/{folder_name}'
    
    df.to_csv(os.path.join(remote_path, file_name))

def find_nearby_systems_circle(df_location, lat, lon, radius):
    """
    Find nearby systems by latitude and longitude in a circular area 
    centered at (lat, lon) with radius as specified.

    Args:
        df_location (pd.DataFrame): location data
        lat (float): latitude
        lon (float): longitude
        radius (float): radius in km
    
    Returns:
        nearby_systems (pd.DataFrame): nearby systems
    """
    # get latitude and longitude of nearby systems
    nearby_systems = df_location[(df_location['latitude_noisy'] - lat)**2 + (df_location['longitude_noisy'] - lon)**2 < radius**2]
    return nearby_systems

def find_nearby_systems_poly(df_location, c1, c2, c3, c4):
    """ 
    Find systems inside a polygonal area specified by the coordinates of 4 corners.
    The input it should have the form
    c1 = (x1, y1) is the bottom left corner 
    c2 = (x2, y2) is the bottom right corner
    c3 = (x3, y3) is the top left corner
    c4 = (x4, y4) is the top right corner

    Args:
        df_location (pd.DataFrame): location data
        c1 (tuple): coordinate of corner 1 (bottom left)
        c2 (tuple): coordinate of corner 2 (bottom right)
        c3 (tuple): coordinate of corner 3 (top left)
        c4 (tuple): coordinate of corner 4 (top right)
    
    Returns:
        nearby_systems (pd.DataFrame): nearby systems
    """
    assert len(c1) == len(c2) == len(c3) == len(c4) == 2, 'Coordinates must be tuples of length 2'
    assert c1[0] < c3[0] and c2[0] < c4[0], 'Coordinates must be in the order: c1, c2, c3, c4'
    assert c1[1] < c2[1] and c3[1] < c4[1], 'Coordinates must be in the order: c1, c2, c3, c4'
    # Create a polygon from the four corner coordinates
    polygon = Polygon([c1, c2, c4, c3])

    # Create a list to store the points (latitude, longitude)
    points = []
    for _, row in df_location.iterrows():
        latitude = row['latitude_noisy']
        longitude = row['longitude_noisy']
        points.append(Point(latitude, longitude))

    # Filter systems that are inside the polygon
    nearby_systems = df_location[[polygon.contains(point) for point in points]]

    return nearby_systems

def stack_dataframe(df_pv, lats_map, longs_map):
    """
    Stacking data frame to include geospatial data
    Use a dictionary mapping to add latitude and longitude to each farm

    Args:
        df_pv (pd.DataFrame): daily production data from day_min to day_max
    
    Returns:
        df_stacked (pd.DataFrame): stacked data frame
    """
    assert 'datetime' in df_pv.columns, 'datetime column not found'

    df_stacked = pd.DataFrame()

    for column in df_pv.drop(columns=['datetime']).columns:
        df = df_pv[['datetime', column]]
        df = df.assign(farm = str(df.columns[-1])).rename(columns = {column:'PV'})
        df_stacked = pd.concat([df_stacked, df])
    
    df_stacked['latitude'] = df_stacked['farm'].map(lats_map)
    df_stacked['longitude'] = df_stacked['farm'].map(longs_map)
    df_stacked['datetime'] = pd.to_datetime(df_stacked['datetime'])
    df_stacked['epoch'] = df_stacked.index

    return df_stacked

# TODO add support for exogenous regressors
# TODO make shapes not adding a new dimension

def create_spatiotemporal_grid(X, Y):
    """
    Create a spatiotemporal grid from a set of spatial points and a set of times.
    It creates an R grid (i.e., a grid of spatial points) and a T grid (i.e., a grid of times).
    The R grid is a matrix of shape (num_times, num_spatial_dims) and the T grid is a vector of shape (num_times,).
    The T grid is the same for all spatial points.

    Args:
        X (np.ndarray): A matrix of shape (num_points, num_spatial_dims + 1) where the first column is the time and the remaining columns are the spatial coordinates.
        Y (np.ndarray): A matrix of shape (num_points, num_outputs) where the first column is the time and the remaining columns are the spatial coordinates.
    
    Returns:
        unique_time (np.ndarray): A vector of shape (num_times, ) containing the unique times.
        R_grid (np.ndarray): A matrix of shape (num_times, num_spatial_dims) containing the spatial coordinates.
        Y_grid (np.ndarray): A matrix of shape (num_times, num_outputs) containing the target values.
    """
    if Y.ndim < 2:
        Y = Y[:, None]
    
    num_spatial_dims = X.shape[1] - 1
    
    sort_args = [X[:, i] for i in range(num_spatial_dims)] + [X[:, 0]]
    sort_ind = np.lexsort(sort_args)
    
    X = X[sort_ind]
    Y = Y[sort_ind]
    
    unique_time = np.unique(X[:, 0])
    unique_space = np.unique(X[:, 1:], axis=0)
    
    grid_shape = (unique_time.shape[0], ) + unique_space.shape
    
    R = np.tile(unique_space, [unique_time.shape[0]] + [1] * num_spatial_dims)
    R_flat = R.reshape(-1, num_spatial_dims)
    
    Y_dummy = np.nan * np.zeros([unique_time.shape[0] * unique_space.shape[0], 1])
    
    time_duplicate = np.tile(unique_time, [unique_space.shape[0], 1]).T.flatten()
    
    X_dummy = np.column_stack([time_duplicate[:, None], R_flat])
    X_all = np.vstack([X, X_dummy])
    Y_all = np.vstack([Y, Y_dummy])
    
    X_unique, ind = np.unique(X_all, axis=0, return_index=True)
    Y_unique = Y_all[ind]
    
    R_grid = X_unique[:, 1:].reshape(grid_shape)
    Y_grid = Y_unique.reshape(grid_shape[:-1] + (1, ))
    
    return unique_time[:, None], R_grid, Y_grid

def remove_nan_systems(r_grid, y):
    """
    Remove systems that have NaN values
    """
    idx = np.argwhere(np.isnan(y))
    idx = np.unique(idx[:, 1])
    r_grid = np.delete(r_grid, idx, axis=1)
    y = np.delete(y, idx, axis=1)
    
    return r_grid, y 

def convert_grid_to_tensor(time, r_grid, y):
    """ 
    Convert spatio temporal grid to tensor

    Args:
        time (np.ndarray): time grid
        r (np.ndarray): spatial grid
        y (np.ndarray): target grid
    
    Returns:
        time (torch.tensor): time grid
        r (torch.tensor): spatial grid
        y (torch.tensor): target grid
    """
    time_tensor = torch.from_numpy(time).float()
    r_grid_tensor = torch.from_numpy(r_grid).float()
    y_tensor = torch.from_numpy(y).float()

    return time_tensor, r_grid_tensor, y_tensor

def extract_time_series(time, y, idx):
    """ 
    Extract time series from spatio temporal grid

    Args:
        time (np.ndarray): time grid
        y (np.ndarray): target grid
    
    Returns:
        time_series (np.ndarray): time series
        y_series (np.ndarray): target series
    """
    time_series = time[:, 0].squeeze(-1)
    y_series = y[:, idx, 0].squeeze(-1)

    return time_series, y_series

def prediction_index(hour, hourly_points, day_max, n_hours):
    """ 
    Get the index of the first data point to predict and
    the index of the last data point to predict.

    Args:
        hour (int): hour of the day
        hourly_points (int): number of data points in an hour
        day_max (int): maximum hour of the day
        n_hours (int): number of hours to predict
    """
    if hour < 8 or hour > 14:
        raise ValueError("Hour argument must be an integer between 8 and 14.")
    
    end_index = (day_max - hour) * hourly_points - 1
    start_index = end_index + n_hours * hourly_points

    return int(start_index), int(end_index)

def train_test_split(X, y, hour, minute_interval=5, day_min=8, day_max=16, n_hours=2):
    """ 
    Splits the data into train and test sets.
    The test set is the last n_hours of the data.

    Args:
        X (torch.tensor): input data
        y (torch.tensor): target data
        hour (int): hour of the day
        minute_interval (int, optional): minute interval between data points. 
        Defaults to 5.
        day_min (int, optional): minimum hour of the day. Defaults to 8.
        day_max (int, optional): maximum hour of the day. Defaults to 16.
        n_hours (int, optional): number of hours to predict. Defaults to 2.
    
    Returns:
        X_train (torch.tensor): train input data
        y_train (torch.tensor): train target data
        X_test (torch.tensor): test input data
        y_test (torch.tensor): test target data
    """
    assert X.shape[0] == y.shape[0], 'X and y must have the same number of rows'

    # number of data points in a day
    daily_points = (day_max - day_min) * 60 // minute_interval 

    hourly_points = daily_points // (day_max - day_min)
  
    start_idx, end_idx = prediction_index(hour, hourly_points, day_max, n_hours)
    print(f'Predicting {n_hours} hours ahead from {hour}:00')

    # split data into train and test sets
    y_train = y[:-start_idx]
    y_test = y[-start_idx:-end_idx]

    if len(X.shape) == 1:
        X_train = X[:-start_idx]
        X_test = X[-start_idx:-end_idx]
    else:
        X_train = X[:-start_idx, :]
        X_test = X[-start_idx:-end_idx, :]

    if torch.cuda.is_available():
        return X_train.cuda(), y_train.cuda(), X_test.cuda(), y_test.cuda()

    return X_train, y_train, X_test, y_test

def cross_val_fold(X, y, n_days, daily_points):
    """
    Splits the data into n_days folds

    Args:
        y (torch.tensor): target data
        periodic_time (torch.tensor): periodic time data
        n_days (int): number of days to use
        daily_points (int): number of points per day
    
    Returns:
        y_list (list): list of target data
        periodic_time_list (list): list of periodic time data
        time (torch.tensor): time data
    """
    interval = int(n_days * daily_points)
   
    x_list = [X[i:i+interval, :] if len(X.shape) > 1 else X[i:i+interval] 
              for i in range(0, len(X), interval)]
    
    y_list = [y[i:i+interval] for i in range(0, len(y), interval)]

    return x_list, y_list


def train_test_split_fold(x_list, y_list, n_hours_pred, daily_points, day_min, day_max):
    """ 
    Split the data into train and test sets
    The test set is a random hour between 8 and 16 - N for all folds 
    (where N is the number of hours to predict). That will avoid 
    discontinuity and facilitates predictions at different times of 
    the day.

    Args: 
        y (list): the list of y values
        periodic_time (list): list of periodic times
        n_hours_pred (int): the number of hours to predict
        daily_points (int): the number of data points per day
    """ 
    assert daily_points % (day_max - day_min) == 0, "daily_points must be divisible by day_max - day_min"
    assert len(x_list) == len(y_list), "y_list and periodic_time_list must have the same length"
    
    y_train = []
    y_test = []
    
    x_train = []
    x_test = []
    
    hourly_data_points = daily_points // (day_max - day_min) 
    
    # indexes going backwards from the end of the array
    min_start_idx = int(n_hours_pred * hourly_data_points)
    max_start_idx = int(daily_points)
    
    for i in range(len(y_list)):

        last_hr_idx = np.random.randint(min_start_idx, max_start_idx)
        start_idx = last_hr_idx + int((n_hours_pred * hourly_data_points))
        
        x_tr = x_list[i][:-start_idx]
        x_te = x_list[i][-start_idx:-last_hr_idx]

        time = torch.arange(0, 100, len(x_tr) + len(x_te)).float()
        
        if len(x_tr.shape) > 1:
            x_tr[:,0] = time[:len(x_tr)]
            x_te[:,0] = time[len(x_tr):]
        else:
            x_tr = time[:len(x_tr)]
            x_te = time[len(x_tr):]

        x_train.append(x_tr)
        x_test.append(x_te)
        
        y_train.append(y_list[i][:-start_idx])
        y_test.append(y_list[i][-start_idx:-last_hr_idx])

    return x_train, y_train, x_test, y_test

def rgetattr(o, k_list):
    """
    Function for deep accessing objects

    For example:
    > rgetattr(o, ['a', 'b']) == o.a.b
    """
    for k in k_list:
        o = getattr(o, k)
    return o

def store_gp_module_parameters(model, n_digits=4, verbose=False):
    """
    Store and print GPyTorch modules in a conveninent way
    (Specifically made for the GP model for tracking purposes
    in weights and biases)
    """
    param_dict = {}
    with torch.no_grad():
        for param_name, param in model.named_parameters():
            # not store variational strategy parameters
            if param_name[:3] == 'var':
                continue
            
            param_name = param_name.replace('raw_', '')
            param = rgetattr(model, param_name.split("."))
            
            if verbose:
                print(
                    f'Name: {param_name:35}\n'
                    f'Value: {param.numpy().round(n_digits)}\n'
                )
                print(f'Real value: {param.numpy().round(n_digits)}\n')
                print('-' * 50)
            param = param.reshape(1)
            param_dict[param_name] = param.numpy().round(n_digits)
            
    return param_dict