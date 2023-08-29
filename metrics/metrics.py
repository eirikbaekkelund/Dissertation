import numpy as np
import torch

def mean_absolute_error(y_pred, y_test):

    if y_pred.shape != y_test.shape:
        raise ValueError('y_pred and y_test must have the same shape')
    
    if isinstance(y_pred, np.ndarray):
       return np.abs(y_pred - y_test)
    elif isinstance(y_pred, torch.Tensor):
        return torch.abs(y_pred - y_test)
   


def get_mean_ci(df):
    mean = df.mean(axis=1)
    lower = df.quantile(0.025, axis=1)
    upper = df.quantile(0.975, axis=1)
    return mean, lower, upper

def inside_ci(lower, upper, y):
    pct_inside = ((y >= lower) & (y <= upper)).sum()
    pct = round((pct_inside / y.shape[0])*100, 2)
    return pct

def nlpd_holt(
    pred_mean: np.ndarray,
    pred_variance: np.ndarray,
    y_test: np.ndarray
):
    """
    Negative log predictive density for Holt model.
    Computes the negative predictive log density normalized by the size of the test data for a Holt model.
    """
    pred_variance[pred_variance == 0] = np.nan
    
    log_density = 0.5 * (np.log(2 * np.pi * pred_variance) + 
                        (y_test - pred_mean)**2 / pred_variance )
    normalized_log_density = log_density / y_test.shape[0]

    return normalized_log_density