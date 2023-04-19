import numpy as np
import pandas as pd
import os
import xarray as xr
import datetime
import time
<<<<<<< HEAD

=======
>>>>>>> e2f5723 (updated loader script, todo fix index of stack)


def load_data(folder_name, file_name):
    """ 
    Load data file from folder 

    Args:
        folder_name (str): folder
        file_name (str): file name
    
    Returns:
        data (xarray): data
    """
    
    assert file_name in os.listdir(folder_name), f'{file_name} not in {folder_name}'
    
    data_path = os.path.join(os.getcwd(), folder_name)
    
    print('==> Loading data')
    tic = time.time()
    if file_name.endswith('.csv'):
        df = pd.read_csv(os.path.join(data_path, file_name))    
    
    elif  file_name.endswith('.netcdf'):
        df = xr.open_dataset(os.path.join(data_path, file_name), engine='h5netcdf').to_dataframe()
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

def save_csv(df, folder_name, file_name):
    """ 
    Save data as csv file

    Args:
        df (pd.DataFrame): data
        folder_name (str): folder
        file_name (str): file name
    """
    data_path = os.path.join(os.getcwd(), folder_name)
    df.to_csv(os.path.join(data_path, file_name))

def find_nearby_systems(df_location, lat, lon, radius):
    """
    Find nearby systems by latitude and longitude
    in a circular area centered at (lat, lon) with radius
    as specified

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
<<<<<<< HEAD
<<<<<<< HEAD

=======
    
>>>>>>> e2f5723 (updated loader script, todo fix index of stack)
=======

>>>>>>> 20a65b3 (loading for stacked systems)
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

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 20a65b3 (loading for stacked systems)
# TODO add support for exogenous regressors
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

<<<<<<< HEAD
=======
>>>>>>> e2f5723 (updated loader script, todo fix index of stack)
=======
>>>>>>> 20a65b3 (loading for stacked systems)

if __name__ == '__main__':
    # Set the parameters
    THRESHOLD = 0.05
    DAY_MIN = 8
    DAY_MAX = 16
    N_SYSTEMS = 10
    # RADIUS = 0.5
    
    # load data
    df_location = load_data('data', 'system_location.csv')
    df_pv = load_data('data', 'pv.netcdf')

    # set index of location data to ss_id
    df_location = set_index(df_location)

    # align pv systems to locations from metadata and the pv data
    df_location, df_pv = align_pv_systems(df_location, df_pv)

    # scale by capacity
    df_pv = scale_by_capacity(df_pv, df_location)

    # drop systems producing over night
    df_pv = drop_night_production(df_pv)

    # get daily production from 8am to 4pm
    df_pv = daily_production(df_pv, 8, 16)

    # remove systems with zero production for more than 5 percent of the time
    df_pv = remove_zero_production(df_pv, 0.05)

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 20a65b3 (loading for stacked systems)
    df_pv['datetime'] = df_pv.index
    df_pv = df_pv.reset_index(drop=True)

    df_location, df_pv = align_pv_systems(df_pv=df_pv,
                                          df_location=df_location)

<<<<<<< HEAD
=======
>>>>>>> e2f5723 (updated loader script, todo fix index of stack)
=======
>>>>>>> 20a65b3 (loading for stacked systems)
    # save pv data & location data
    save_csv(df_pv, 'data', 'pv_data_clean.csv')
    save_csv(df_location, 'data', 'location_data_clean.csv')

    # set datetime to column and reindex to range from 0 to len(df_pv)
    df_pv['datetime'] = df_pv.index
    df_pv = df_pv.reset_index(drop=True)

    lats, longs = get_location_maps(df_location, N_SYSTEMS)

    print(f"Location Farms: {list(lats.keys())}\n")
    print(f"PV Farms: {list(df_pv.drop(columns=['datetime']).columns)}\n")

    print(f"Location DF dtypes: {df_location.dtypes}\n")
    print(f"PV DF dtypes: {df_pv.dtypes}\n")

<<<<<<< HEAD
<<<<<<< HEAD
    # list of strings of pv farms
    pvs = [float(pv) for pv in lats.keys()]
    df_pv = df_pv[['datetime'] + pvs]
=======
    df_pv = df_pv[['datetime'] + list(lats.keys())]
>>>>>>> e2f5723 (updated loader script, todo fix index of stack)
=======
    # list of strings of pv farms
    pvs = [float(pv) for pv in lats.keys()]
    df_pv = df_pv[['datetime'] + pvs]
>>>>>>> 20a65b3 (loading for stacked systems)

    df_stacked = stack_dataframe(df_pv, lats, longs)

    save_csv(df_stacked, 'data', 'pv_data_stacked.csv')
