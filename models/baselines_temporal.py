import numpy as np
from statsmodels.tsa.api import (VAR, 
                                 ExponentialSmoothing, 
                                 SimpleExpSmoothing, 
                                 Holt)
from data.utils import get_hourly_points, get_daily_points

class Persistence():

    def predict(self, y, n_steps):
        y_pred = np.zeros((n_steps, y.shape[1])) if len(y.shape) > 1 else np.zeros(n_steps)
        y_pred[:] = y[-1]
        return y_pred

class YesterdayForecast():
    def __init__(self, day_min, day_max, minute_interval):
        
        self.n_hourly_points = get_hourly_points(day_min, day_max, minute_interval)
        self.daily_points = get_daily_points(day_min, day_max, minute_interval)
    
    def predict(self, y, n_hours=2):
        yday_start = self.daily_points
        pred_length = int(self.daily_points - self.n_hourly_points * n_hours)
        return y[-yday_start:-pred_length]

class HourlyAverage():
    def __init__(self, day_min, day_max, minute_interval):
        self.minute_interval = minute_interval
        self.n_hourly_points = get_hourly_points(day_min, day_max, minute_interval)

    def predict(self, y, n_hours=2):
        """ 
        Use hourly average to predict n_hours ahead.
        It should predict by doing 

        y_pred = 1 / T * \sum_{t=1}^T y_{t + \delta t - I*t} 
        where T is the number of points per hour and I is the
        minute interval of the data.

        Args:
            y_train (np.ndarray): training data of shape (N, T)
            where N is the number of data points and T is the
            number of tasks

            n_hours (int): number of hours to predict ahead
        """
        n_preds = int(n_hours * self.n_hourly_points)
        y_pred_means = y[-self.n_hourly_points:].mean(axis=0)
        y_preds = np.tile(y_pred_means, (n_preds, 1))
        
        return y_preds

def fit_var(y_train, pred_points):
  
    if np.isnan(y_train).any():
        # remove nan
        y_train = y_train[~np.isnan(y_train).any(axis=1)]
    var = VAR(y_train)
    fitted_model = var.fit()
    lag_order = fitted_model.k_ar
    y_pred = fitted_model.forecast(y_train[-lag_order:], pred_points)
    # clip predictions to 0 and 1
    y_pred = np.clip(y_pred, 0, 1)
    return y_pred

def fit_exp(y_train, pred_points):
   
    if np.isnan(y_train).any():
        y_train = y_train[~np.isnan(y_train).any(axis=0)]
        if y_train.shape[0] == 0:
            return np.full(pred_points, np.nan)
    model = ExponentialSmoothing(y_train,
                                 seasonal_periods=96,
                                 seasonal='add',
                                 initialization_method='estimated')
    fitted_model = model.fit()
    y_pred = fitted_model.forecast(pred_points)
    y_pred = np.clip(y_pred, 0, 1)
    return y_pred, fitted_model

def fit_simple_exp(y_train, pred_points):

    if np.isnan(y_train).any():
        y_train = y_train[~np.isnan(y_train).any(axis=0)]
        if y_train.shape[0] == 0:
            return np.full(pred_points, np.nan)
    model = SimpleExpSmoothing(y_train, initialization_method='estimated')
    fitted_model = model.fit()
    y_pred = fitted_model.forecast(pred_points)
    y_pred = np.clip(y_pred, 0, 1)
    return y_pred, fitted_model

def var_exp_simulation(model_fit, y_pred, n_pred=24):
    var = model_fit.simulate(nsimulations=n_pred, anchor='end', repetitions=1000).var(axis=1)
    
    var_lower = (y_pred**2 / 4)
    var_upper = (1 - y_pred)**2 / 4
    
    var = np.maximum(var, var_lower)
    var = np.minimum(var, var_upper)
    return var



        
