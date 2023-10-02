import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from data.generator import PVWeatherGenerator
from typing import Optional
from data.utils import get_lat_lon_col_names
from data.utils import train_test_split
from scipy.signal import savgol_filter


class PVDataLoader(Dataset):
    """ 
    Custom dataset for the PV data to use from the CV folds

    Args:
        X (torch.tensor): list of input data
        y (torch.tensor): list of target data
    """
    def __init__(self, X : list, y : list):
        assert len(X) == len(y), 'X and y must have the same length'
        self.x = X
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)

class PVWeatherLoader(PVWeatherGenerator, Dataset):
    """ 
    In the __init__ method, create a list of all the unique lat lon pairs from
    the df, then in the __getitem__ method, return the X and y values for each system
    """
    def __init__(self, 
                 coords : tuple, 
                 radius : float = 1,
                 day_init : int = 0,
                 n_systems : int = 15, 
                 n_days : int = 5,
                 minute_interval : int = 60,
                 day_min : int = 8, 
                 day_max : int = 15,
                 folder_name : str = 'pv_data',
                 file_name : str = 'pv_and_weather.csv',
                 distance_method : str = 'circle',
                 season : Optional[str] = None,
                 drop_nan : bool = True,
                 x_cols : Optional[list] =['global_rad:W', 'diffuse_rad:W',
                                            'effective_cloud_cover:octas',
                                            'relative_humidity_2m:p', 't_2m:C'],
                 y_col : Optional[str] ='PV'
                ):
        
        super().__init__(coords=coords, 
                         radius=radius,
                         day_init=day_init,
                         n_systems=n_systems, 
                         n_days=n_days,
                         minute_interval=minute_interval,
                         day_min=day_min, 
                         day_max=day_max,
                         folder_name=folder_name,
                         file_name=file_name,
                         distance_method=distance_method,
                         season=season,
                         drop_nan=drop_nan,
                         
                        )
        
        # create list of all X and y values for each system based on lat lon pairs
        self.create_X_y(x_cols=x_cols, y_col=y_col)

    def create_X_y(self, x_cols : list, y_col : str):
        """
        Create list of all X and y values for each system based on lat lon pairs

        Args:
            x_cols (list): list of column names to use as X
            y_col (str): column name to use as y
        """
        
        X = []
        y = []
        
        lat_col, lon_col = get_lat_lon_col_names(self.df)
        
        for i in range(len(self.unique_lat_lon)):
            lat, lon = self.unique_lat_lon.iloc[i]
            df = self.df[(self.df[lat_col] == lat) & (self.df[lon_col] == lon)]
            X.append(df[x_cols].values)
            y.append(df[y_col].values)
        
        self.X = X
        self.y = y

    def __getitem__(self, index):
        
        _X = self.X[index][self.start_idx:self.end_idx]
        X = torch.tensor(_X, dtype=torch.float32)

        _y = self.y[index][self.start_idx:self.end_idx]
        y = torch.tensor(_y, dtype=torch.float32)
        
        return X, y
    
    def __len__(self):
        return len(self.X)
    
# designed to work with the PVWeatherGenerator dataframe 
# also designed to run the experiment due to its train test split return

class SystemLoader(Dataset):
    """ 
    SystemLoader is a custom dataset for the PV data to use from the folds.
    Loops through the systems either in a combined set of X, y, and task indices
    that is split into train and test sets by a concetanation of the X and y values,
    or by individual systems that are split into train and test sets by the task indices.

    Loop through the loader to get the X, y, and task indices for concatenation or loop through 
    the individual systems by running:
    >>> # first loop through the loader of all systems
    >>> for X_tr, Y_tr, X_te, Y_te, T_tr, T_te in loader:
    >>> # then loop through the individual systems
    >>>     for i in range(len(T_te.unique())):
    >>>         x_tr, y_tr, x_te, y_te = loader.train_test_split_individual(i)

    Args:
        df (pd.DataFrame): dataframe of the PV data
        train_interval (int): number of days to use for train-test split
        x_cols (list, optional): list of column names to use as X
        season (str, optional): season to use for the data. Defaults to None.
    """
    def __init__(
            self,
            df : pd.DataFrame,
            train_interval : int,
            x_cols : Optional[list] =['global_rad:W', 
                                      'diffuse_rad:W', 
                                      'effective_cloud_cover:octas',
                                      'relative_humidity_2m:p', 't_2m:C',
                                      'wind_speed_10m:ms'],
            season : Optional[str] = None,
            n_hours_pred : int = 6
    ):
        super().__init__()
        if season is not None:
            assert season in ["winter", "summer", "spring", "fall"], \
                            'season must be one of "winter", "summer", "spring", "fall"'
            df = df[df["season"] == season]
            # drop season column
            df = df.drop(columns=["season"], axis=1)
        
        # set task indices to the unique latitude and longitude pairs in X
        task_indices = torch.ones(len(df), dtype=torch.long)
        lat, lon = get_lat_lon_col_names(df)
        unique_lat_lon = df[[lat, lon]].drop_duplicates().values
        
        for i, lat_lon in enumerate(unique_lat_lon):
            task_indices[(df[lat] == lat_lon[0]) & (df[lon] == lat_lon[1])] = \
            task_indices[(df[lat] == lat_lon[0]) & (df[lon] == lat_lon[1])] * i
        
        self.X = torch.tensor(df[x_cols].values, dtype=torch.float64)
        self.y = torch.tensor(df['PV'].values, dtype=torch.float64)
        
        self.tasks = task_indices
        self.n_systems = len(unique_lat_lon)
        self.train_interval = train_interval
        self.n_hours_pred = n_hours_pred
        
        # index to start at
        self.start = 0
        self.end = self.train_interval
    
    def update_index(self):
        self.start += self.train_interval
        self.end = self.start + self.train_interval

    def __len__(self):
        return len(self.tasks[self.tasks == 0]) // self.train_interval
    
    def slice_data(self, i):
        """
        Slice the data for the ith system
        """
        x = self.X[self.tasks == i][self.start:self.end]
        # normalize x
        for j in range(x.shape[1]):
            x[:, j] = (x[:, j] - x[:, j].mean()) / x[:, j].std()
        
        time_dim_x = torch.linspace(0, 1, x.shape[0], dtype=torch.float64)
        x = torch.cat((x, time_dim_x.unsqueeze(1)), dim=-1)
    
        t = self.tasks[self.tasks == i][self.start:self.end]
        y = self.y[self.tasks == i][self.start:self.end]

        if y.max() >= 1:
            y[y >= 1] = 1 - 1e-6
        if y.min() <= 0:
            y[y <= 0] = 1e-6

        return x.float(), y, t

    def train_test_split_region(self):
        """
        Split the data into train and test sets using all systems.
        """
        X_train, X_test = [], []
        Y_train, Y_test = [], []
        T_train, T_test = [], []

        # get a random hour between 8 and 14
        hour = np.random.randint(8, 16 - self.n_hours_pred)
        self.hour = hour
        for i in range(self.n_systems):
            x, y, t = self.slice_data(i)
           
            x_train, y_train, x_test, y_test = train_test_split(
                                                X=x,
                                                y=y, 
                                                hour=hour,
                                                n_hours=self.n_hours_pred
                                            )
            n_tr, n_te = len(x_train), len(x_test)
            task_train, task_test = t[:n_tr], t[n_tr:n_tr+n_te]

            # run savgol filter on input data except time dimension
            if len(x_train) > 12:
                x_train[:, :-1] = torch.tensor(savgol_filter(x_train[:, :-1], 
                                                    window_length=12, polyorder=3, axis=0), 
                                                dtype=torch.float32)
                x_test[:, :-1] = torch.tensor(savgol_filter(x_test[:, :-1], 
                                                window_length=12, polyorder=3, axis=0), 
                                            dtype=torch.float32)
            X_train.append(x_train)
            Y_train.append(y_train)
            T_train.append(task_train)
            X_test.append(x_test)
            Y_test.append(y_test)
            T_test.append(task_test)

        self.x_train = torch.cat(X_train, dim=0)
        self.y_train = torch.cat(Y_train, dim=0)
        self.task_train = torch.cat(T_train, dim=0)
        
        self.x_test = torch.cat(X_test, dim=0)
        self.y_test = torch.cat(Y_test, dim=0)
        self.task_test = torch.cat(T_test, dim=0)

    
    def train_test_split_individual(self, i):
        """
        Split the data into train and test sets using the ith system.
        """

        x_train = self.x_train[self.task_train == i]
        y_train = self.y_train[self.task_train == i]
        x_test = self.x_test[self.task_test == i]
        y_test = self.y_test[self.task_test == i]
        
        return x_train, y_train, x_test, y_test
    
    def reset(self):
        self.start = 0
        self.end = self.train_interval
    
    def __getitem__(self, idx):
       
        self.train_test_split_region()

        if idx == len(self):
            # stop iteration at the end of the dataset
            self.reset()
            raise StopIteration
        elif self.x_train.shape[0] == 0:
            self.reset()
            raise StopIteration
        # if break is called, reset the index

        
        self.update_index()

        return self.x_train, self.y_train, self.x_test, \
               self.y_test, self.task_train, self.task_test
