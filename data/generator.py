import torch
from data.utils import *
from typing import Optional

class PVDataGenerator:
    """
    Data loader for the Temporal GP model without weather data.
    Uses the data from the pv_data_clean.csv file and the pv_data_location.csv file.

    Args:
        n_days (int): number of days to use
        minute_interval (int): interval between data points in minutes
        n_systems (int): number of systems to use
        radius (float): radius in km
        coords (tuple): coordinates of the center of the area
        day_min (int): minimum hour of the day to use
        day_max (int): maximum hour of the day to use
        folder_name (str): name of the folder where the data is stored
        file_name_pv (str): name of the file where the pv data is stored
        file_name_location (str): name of the file where the location data is stored
        distance_method (str): method to use to find nearby systems, either 'circle' or 'poly'
        drop_nan (bool): whether to drop systems with NaN values
    """
    def __init__(self,                 
                 coords : tuple, 
                 radius : float = 1,
                 day_init : int = 0,
                 n_systems : int = 15, 
                 n_days : int = 5,
                 minute_interval : int = 5,
                 day_min : int = 8, 
                 day_max : int = 16,
                 folder_name : str = 'pv_data',
                 file_name_pv : str = 'pv_data_clean.csv',
                 file_name_location : str = 'location_data_clean.csv',
                 distance_method : str = 'circle',
                 season : Optional[str] = None,
                 drop_nan : bool = True):
        
        assert distance_method in ['circle', 'poly'], 'distance_method must be either circle or poly'
        assert day_min < day_max, 'day_min must be smaller than day_max'
        assert day_min >= 0 and day_max <= 24, 'day_min and day_max must be between 0 and 24'
        assert n_systems > 0, 'n_systems must be greater than 0'
        assert n_days > 0, 'n_days must be greater than 0'
        assert minute_interval > 0, 'minute_interval must be greater than 0'
        
        if distance_method == 'circle':
            assert radius > 0, 'radius must be greater than 0'
            assert len(coords) == 2, 'coords must be a tuple of length 2 when using circle method'
        else:
            assert len(coords) == 4, 'coords must be a tuple of length 4 when using poly method'
        if season is not None:
            assert season in ['winter', 'spring', 'summer', 'fall'], 'season must be one of winter, spring, summer, autumn'
        # load data
        df_pv = load_data(folder_name=folder_name, file_name=file_name_pv)
        df_location = load_data(folder_name=folder_name, file_name=file_name_location)
        
        # set index
        df_location = set_index(df_location=df_location)
        date_time = df_pv['datetime']
        
        # get systems in the area 
        if distance_method == 'circle':
            systems = find_nearby_systems_circle(df_location=df_location, 
                                            lat=coords[0], 
                                            lon=coords[1], 
                                            radius=radius)
        else:
            systems = find_nearby_systems_poly(df_location, *coords)
        
        # update number of systems to use by the number of systems in the area
        systems, pv_series = align_pv_systems(df_location=systems, df_pv=df_pv)
        lats, longs = get_location_maps(df_location=systems, n_systems=n_systems)

        # scale pv data by making it 0-1 by dividing by max value in each column
        pv_series = pv_series / pv_series.max()
        pv_series = pv_series[::minute_interval // 5] # time interval is 5 minutes in the data

        with pd.option_context('mode.chained_assignment', None):
            pv_series['datetime'] = date_time
        
        # update index of pv_series
        if season is not None: 
            pv_series = filter_by_season(df=pv_series, season=season)
            
        start_idx, end_idx = start_end_index(day_min=day_min, 
                                             day_max=day_max, 
                                             minute_interval=minute_interval, 
                                             n_days=n_days,
                                             day_init=day_init)

        # get relevant sample of pv series
        date_time = pv_series['datetime'].iloc[start_idx:end_idx]
        pv_series = pv_series.iloc[start_idx:end_idx, :n_systems]
        pv_series['datetime'] = date_time

        # create stack of all systems
        pv_series = stack_dataframe(df_pv=pv_series, lats_map=lats, longs_map=longs)

        self.pv_series = pv_series.copy()

        # save_csv(df=pv_series, folder_name=folder_name, file_name='pv_data_stack.csv')
        
        X = pv_series[['epoch', 'latitude', 'longitude']].values
        y = pv_series['PV'].values

        time, r_grid, y = create_spatiotemporal_grid(X, y)
        
        y = y.squeeze()
        time = time.squeeze()

        # TODO add functionality to interval all data by n_days
        # TODO create a cross_val_fold method

        if drop_nan:
            r_grid, y = remove_nan_systems(r_grid=r_grid, y=y)

        self.time_tensor, self.r_grid_tensor, self.y_tensor = convert_grid_to_tensor(time=time, r_grid=r_grid, y=y)

        self.day_min = day_min
        self.day_max = day_max
        self.minute_interval = minute_interval
        
    def get_time_series(self, include_periodic : bool = False):

        y = torch.clamp(self.y_tensor, min=1e-7, max=1 - 1e-7)
        
        if include_periodic:
            periodic_time = periodic_mapping(self.time_tensor, day_min=self.day_min, day_max=self.day_max, minute_interval=self.minute_interval)
            X = torch.stack([time_tensor, periodic_time], dim=-1)
        else:
            X = self.time_tensor
        
        time_tensor = torch.linspace(0, 100, len(self.time_tensor)).float()
        
        if torch.cuda.is_available():
            return self.time_tensor.cuda(), self.y_tensor.cuda()
        
        return X, y

class PVWeatherGenerator:
    """
    Data loader for the Temporal GP model with weather data
    sampled at hourly intervals.
    Uses the data from the pv_and_weather.csv file 

    Args:
        coords (tuple): coordinates of the center of the area
        radius (float): radius 
        n_systems (int): number of systems to use
        n_days (int): number of days to use
        minute_interval (int): interval between data points in minutes
        day_min (int): minimum hour of the day to use
        day_max (int): maximum hour of the day to use
        folder_name (str): name of the folder where the data is stored
        file_name (str): name of the file where the data is stored
        distance_method (str): method to use to find nearby systems, either 'circle' or 'poly'
        drop_nan (bool): whether to drop systems with NaN values
    """
    def __init__(self, 
                 coords : tuple, 
                 radius : float = 1,
                 day_init : int = 0,
                 n_systems : int = 15, 
                 n_days : int = 5,
                 minute_interval : int = 5,
                 day_min : int = 8, 
                 day_max : int = 16,
                 folder_name : str = 'pv_data',
                 file_name : str = 'pv_and_weather.csv',
                 distance_method : str = 'circle',
                 season : Optional[str] = None,
                ):
        
        assert distance_method in ['circle', 'poly', 'all'], 'distance_method must be either circle or poly'
        assert day_min < day_max, 'day_min must be smaller than day_max'
        assert day_min >= 0 and day_max <= 24, 'day_min and day_max must be between 0 and 24'
        assert n_systems > 0, 'n_systems must be greater than 0'
        assert n_days > 0, 'n_days must be greater than 0'
        assert minute_interval > 0, 'minute_interval must be greater than 0'
    
        if distance_method == 'circle':
            assert radius > 0, 'radius must be greater than 0'
            assert len(coords) == 2, 'coords must be a tuple of length 2 when using circle method'
        elif distance_method == 'poly':
            assert len(coords) == 4, 'coords must be a tuple of length 4 when using poly method'
        
        df = load_data(folder_name=folder_name, file_name=file_name)
        
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        
        if season is not None:
            df = filter_by_season(df=df, season=season)
        
        df.set_index('datetime', inplace=True)
        
        if distance_method == 'circle':
            df = find_nearby_systems_circle(df, lat=coords[0], lon=coords[1], radius=radius)
        elif distance_method == 'poly':
            df = find_nearby_systems_poly(df, *coords)
      
        lat_col, lon_col = get_lat_lon_col_names(df)
        
        self.unique_lat_lon = df[[lat_col, lon_col]].drop_duplicates()
        self.unique_lat_lon = self.unique_lat_lon[:n_systems]
        
        df = df[df[lat_col].isin(self.unique_lat_lon[lat_col]) & df[lon_col].isin(self.unique_lat_lon[lon_col])]
        
        # smooths out the linear interpolation of the data
        # TODO consider to do this prior to saving the files
       
        self.df = df.copy()
    
        self.start_idx, self.end_idx = start_end_index(
                                        day_init=day_init,
                                        day_min=day_min, 
                                        day_max=day_max, 
                                        minute_interval=minute_interval, 
                                        n_days=n_days)
