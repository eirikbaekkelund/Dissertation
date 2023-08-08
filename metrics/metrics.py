import numpy as np

def mean_absolute_error(y_pred, y_test):
    if y_pred.shape != y_test.shape:
        raise ValueError('y_pred and y_test must have the same shape')
    mae = np.abs(y_pred - y_test)
    return mae

def nlpd(y_pred, y_test, y_std):
    if y_pred.shape != y_test.shape:
        raise ValueError('y_pred and y_test must have the same shape')
    # calculate metrics
    nlpd = 0.5 * np.log(2 * np.pi * y_std**2) + 0.5 * ((y_pred - y_test) / y_std)**2
    return nlpd

def get_mean_ci(df):
    mean = df.mean(axis=1)
    lower = df.quantile(0.025, axis=1)
    upper = df.quantile(0.975, axis=1)
    return mean, lower, upper

def inside_ci(lower, upper, y):
    pct_inside = ((y >= lower) & (y <= upper)).sum()
    pct = round((pct_inside / y.shape[0])*100, 2)
    return pct

def log_score_function(actual, predicted, var):
    var[var == 0] = np.nan
    const = np.log(2 * np.pi)
    L = 0.5 * np.nansum(np.log(var) + (actual - predicted)**2 / var + const, axis=1)
    return L