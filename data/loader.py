import torch
from torch.utils.data import Dataset
from data.generator import PVWeatherGenerator
from typing import Optional
from data.utils import get_lat_lon_col_names


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
    