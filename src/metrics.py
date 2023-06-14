import torch
import numpy as np

def rmse_np(x,y):
    return np.sqrt(np.mean((x-y)**2))

def mae_np(x,y):
    return np.mean(np.abs(x-y))

def mape_np(x,y):
    return np.mean(np.abs((x-y)/x))

def smape_np(x,y):
    return np.mean(np.abs((x-y)/(x+y)))

def nrmse_np(x,y):
    return np.sqrt(np.mean((x-y)**2))/np.std(y)

def nmae_np(x,y):
    return np.mean(np.abs(x-y))/np.std(y)

def nmse_np(x,y):
    return np.mean((x-y)**2)/np.var(y)

def rmse_torch(x,y):
    return torch.sqrt(torch.mean((x-y)**2))

def mae_torch(x,y):
    return torch.mean(torch.abs(x-y))

def mape_torch(x,y):
    return torch.mean(torch.abs((x-y)/x))

def smape_torch(x,y):
    return torch.mean(torch.abs((x-y)/(x+y)))

def nrmse_torch(x,y):
    return torch.sqrt(torch.mean((x-y)**2))/torch.std(y)

def nmae_torch(x,y):
    return torch.mean(torch.abs(x-y))/torch.std(y)

def nmse_torch(x,y):
    return torch.mean((x-y)**2)/torch.var(y)

