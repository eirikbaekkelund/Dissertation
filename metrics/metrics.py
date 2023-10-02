import numpy as np
import torch
from gpytorch.metrics import negative_log_predictive_density as nlpd

def mean_absolute_error(y_pred, y_test):
    
    if y_pred.shape != y_test.shape:
        raise ValueError('y_pred and y_test must have the same shape')
    
    if isinstance(y_pred, np.ndarray):
       return np.abs(y_pred - y_test)
    elif isinstance(y_pred, torch.Tensor):
        return torch.abs(y_pred - y_test)
   
def get_mean_ci(df):
    median = df.median(axis=1)
    lower = df.quantile(0.025, axis=1)
    upper = df.quantile(0.975, axis=1)
    return median, lower, upper

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

def neg_log_pred(dist, y_test):
    """
    Negative log predictive density for GP model.
    Computes the negative predictive log density normalized by the size of the test data for a GP model.
    """
    return nlpd(dist, y_test)

def neg_log_pred_hadamard(dist, y_test):
    """
    Negative log predictive density for Hadamard model.
    Computes the negative predictive log density rescaled by the number of tasks for a Hadamard model.
    Otherwise, the negative predictive log density = normalized nlpd / num_tasks * n_points
    """
    
    return nlpd(dist, y_test).t() 