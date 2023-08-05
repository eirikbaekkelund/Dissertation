from abc import ABC, abstractmethod
from statsmodels.tsa.api import AutoReg, ARIMA
from data.utils import get_hourly_data_points, get_daily_data_points

class BaselineForecast(ABC):
    def __init__(self, y_train):
        self.y_train = y_train
    
    @abstractmethod
    def fit(self):
        """ 
        Fit the model
        """
        pass

    @abstractmethod
    def predict(self, n_steps):
        """ 
        Predict n_steps ahead
        """
        pass

class PersistenceForecast(BaselineForecast):
   
    def fit(self):
        pass

    def predict(self, n_steps):
        return self.y_train[-1].repeat(n_steps)

class YesterdayForecast(BaselineForecast):
    def __init__(self, y_train, minute_interval, day_min, day_max):
        super().__init__(y_train)
        self.n_hourly_points = get_hourly_data_points(y_train, minute_interval, day_min, day_max)
        self.daily_points = get_daily_data_points(y_train, minute_interval, day_min, day_max)
    
    def fit(self):
        pass

    def predict(self, n_hours):
        start_idx = self.daily_points
        end_idx = start_idx + self.n_hourly_points
        return self.y_train[start_idx:end_idx]
        
        
