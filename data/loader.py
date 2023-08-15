import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from data.generator import PVWeatherGenerator
from typing import Optional
from data.utils import get_lat_lon_col_names
from data.utils import train_test_split


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
                                            'relative_humidity_2m:p', 't_2m:C',
                                            'wind_speed_10m:ms'],
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
    def __init__(
            self,
            X : pd.DataFrame,
            y : pd.DataFrame,
            train_interval : int,
            season : Optional[str] = None,
    ):
        super().__init__()
        if season is not None:
            assert season in ["winter", "summer", "spring", "fall"], \
                            'season must be one of "winter", "summer", "spring", "fall"'
            X = X[X["season"] == season]
            y = y[X["season"] == season]
            # drop season column
            X = X.drop(columns=["season"], axis=1)
        
        # set task indices to the unique latitude and longitude pairs in X
        task_indices = torch.ones(len(X), dtype=torch.long)
        lat, lon = get_lat_lon_col_names(X)
        unique_lat_lon = X[[lat, lon]].drop_duplicates().values
        
        for i, lat_lon in enumerate(unique_lat_lon):
            task_indices[(X[lat] == lat_lon[0]) & (X[lon] == lat_lon[1])] = \
            task_indices[(X[lat] == lat_lon[0]) & (X[lon] == lat_lon[1])] * i
        
        self.X = torch.tensor(X.values, dtype=torch.float64)
        self.y = torch.tensor(y.values, dtype=torch.float64)
        
        self.tasks = task_indices
        self.n_systems = len(unique_lat_lon)
        self.train_interval = train_interval
        
        # index to start at
        self.start = 0
        self.end = self.train_interval
    
    def set_index(self):
        self.start += self.train_interval
        self.end = self.start + self.train_interval

    def __len__(self):
        return len(self.tasks[self.tasks == 0]) // self.train_interval
    
    def slice_data(self, i):
        x = self.X[self.tasks == i][self.start:self.end]
        t = self.tasks[self.tasks == i][self.start:self.end]
        y = self.y[self.tasks == i][self.start:self.end]

        return x, y, t

    def train_test_split_region(self):
        X_train, X_test = [], []
        Y_train, Y_test = [], []
        T_train, T_test = [], []

        # get a random hour between 8 and 14
        hour = np.random.randint(8, 14)
        
        for i in range(self.n_systems):
            x, y, t = self.slice_data(i)
           
            x_train, y_train, x_test, y_test = train_test_split(X=x, y=y, hour=hour,n_hours=2)
            n_tr, n_te = len(x_train), len(x_test)
            task_train, task_test = t[:n_tr], t[n_tr:n_tr+n_te]

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
        x_train = self.x_train[self.task_train == i]
        y_train = self.y_train[self.task_train == i]
        x_test = self.x_test[self.task_test == i]
        y_test = self.y_test[self.task_test == i]
        
        return x_train, y_train, x_test, y_test

    def __getitem__(self, idx):
       
        self.train_test_split_region()

        if idx == len(self) + 1:
            # stop iteration at the end of the dataset
            raise StopIteration
        
        self.set_index()

        return self.x_train, self.y_train, self.x_test, \
               self.y_test, self.task_train, self.task_test
