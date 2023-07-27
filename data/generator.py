import torch
from utils import *

class PVDataGenerator:
    """
    Data loader for the Temporal GP model

    Args:
        n_days (int): number of days to use
        minute_interval (int): interval between data points in minutes
        n_systems (int): number of systems to use
        radius (float): radius in km
        coords (tuple): coordinates of the center of the area
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
                 file_name_location : str = 'pv_data_location.csv',
                 distance_method : str = 'circle',
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
       
        # load data for pv and location
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
        pv_series = pv_series[::minute_interval//5] # time interval is 5 minutes in the data

        n_daily_points = (day_max - day_min) * 60 // minute_interval
        n_samples = int(n_daily_points * n_days)
        
        # get relevant sample of pv series
        pv_series = pv_series.iloc[n_samples*day_init:n_samples*(day_init + 1), :n_systems]

        # update index of pv_series
        with pd.option_context('mode.chained_assignment', None):
            pv_series['datetime'] = date_time

        # create stack of all systems
        pv_series = stack_dataframe(df_pv=pv_series, lats_map=lats, longs_map=longs)

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
        self.time_tensor = torch.arange(0, len(self.time_tensor)).float()
        self.day_min = day_min
        self.day_max = day_max
        self.minute_interval = minute_interval
        
    def get_time_series(self):

        if torch.cuda.is_available():
            return self.time_tensor.cuda(), self.y_tensor.cuda()
        
        y = torch.clamp(self.y_tensor, min=1e-7, max=1 - 1e-7)
        
        periodic_time = periodic_mapping(self.time_tensor, day_min=self.day_min, day_max=self.day_max, minute_interval=self.minute_interval)
        X = torch.stack([self.time_tensor, periodic_time], dim=-1)

        return X, y