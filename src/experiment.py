import numpy as np
import torch
import gpytorch
import time
from src.data_loader import PVDataLoader
from gpytorch.likelihoods import GaussianLikelihood
from src.models import BetaGP, GaussianGP, ExactGPModel
from src import data_loader as dl
from src.metrics import rmse_np, mae_np 

# set seed
SEED = 42
np.random.seed(SEED)

# I know it is ugly, it is for convenience sake :)

def run_gp_experiments(inputs : dict,
                   num_inducing_points : list,
                   days : list = np.arange(0, 200, 10),
                   models : list = ['BetaSVI', 'GaussianSVI', 'GaussianExact'],
                   n_svi : int = 10,
                   names : list = ['cholesky', 'mean_field'],
                   hyper_params : dict = {'lr': 0.1, 
                                          'n_iter': 100, 
                                          'optim' : torch.optim.Adam, 
                                          'device': torch.device('cpu')},
                   metrics : list = [rmse_np, mae_np]
                   ):
    """
    Run an experiment over a grid of configurations &
    models and save the results to a txt file.
    The SVI models are trained several times to account
    for the stochasticity of the optimization similar
    to that of neural networks as the optimization is
    non-convex.

    Args:
        inputs (dict): dictionary of inputs to the GP model
        num_inducing_points (list): list of number of inducing points
        days (list, optional): list of number of days to train on. Defaults to np.arange(0, 200, 10).
        models (list, optional): list of models to train. Defaults to ['BetaSVI', 'GaussianSVI', 'GaussianExact'].
        n_svi (int, optional): number of times to train the SVI models. Defaults to 5.
        names (list, optional): list of names of the variational distributions. Defaults to ['cholesky', 'mean_field'].
        hyper_params (dict, optional): dictionary of hyperparameters for the optimizer. Defaults to {'lr': 0.1, 'n_iter': 100, 'optim' : torch.optim.Adam, 'device': torch.device('cpu')}.
        metrics (list, optional): list of metrics to evaluate the model. Defaults to [rmse, mae].
    """
    
    config = {'type': 'stochastic',
              'mean_init_std': 1e-2}
    
    results = {'GaussianSVI': None,
                'BetaSVI': None,
                'GaussianExact': None
                }
    
    time_total_start = time.time()

    for n in names:
        for n_ind in num_inducing_points:
            for d in days:
                for model in models:
                    # add keys and values to config
                    config['name'] = n
                    config['num_inducing_points'] = n_ind
                    config['n_days'] = d
                    config['model'] = model
                    inputs['config'] = config

                    # run experiment
                    if model == 'BetaSVI':
                        model = BetaGP(**inputs)
                    elif model == 'GaussianSVI':
                        model = GaussianGP(**inputs)
                    elif model == 'GaussianExact':
                        inputs['likelihood'] = GaussianLikelihood() 
                        model = ExactGPModel(**inputs)
                        # delete likelihood from inputs
                        del inputs['likelihood']
                    
                    # train model and predict
                    # TODO random initialization of hyperparameters for each run
                    rmse_list = []
                    mae_list = []
                    
                    print(f'Running experiment with {model} for {n_svi} iterations')
                    time_start_fit = time.time()
                    for _ in range(n_svi):
                        model.fit(**hyper_params)
                        print(f'Model: {model}')
                        preds = model.predict(inputs['x_test'], device=hyper_params['device'])
                        
                        if isinstance(model.likelihood, gpytorch.likelihoods.GaussianLikelihood):
                            preds = preds.mean
                        else:
                            preds = preds.mean.mean(axis=0)
                        
                        rmse_list.append(rmse_np(inputs['y_test'], preds))
                        mae_list.append(mae_np(inputs['y_test'], preds))
                    
                        # TODO add uncertainty metrics 
                    time_end_fit = time.time()
                    print(f'Finished training in {time_end_fit - time_start_fit:.2f} seconds')
                    results[model] = {'rmse' : (np.mean(rmse_list), np.std(rmse_list)),
                                      'mae' : (np.mean(mae_list), np.std(mae_list))}


if __name__ == '__main__':
   
    DAY_MIN = 8
    DAY_MAX = 16
    N_DAYS = 5
    MINUTE_INTERVAL = 5
    DAILY_DATA_POINTS = (DAY_MAX - DAY_MIN) * 60 / MINUTE_INTERVAL
    N_SYSTEMS = 30
    RADIUS = 0.75
    COORDS = (55.55074, -4.3278)
    IDX = 6
   
    loader = PVDataLoader(n_days=N_DAYS,
                    day_init=10,
                    n_systems=N_SYSTEMS,
                    radius=RADIUS,
                    coords=COORDS,
                    minute_interval=MINUTE_INTERVAL,
                    day_min=DAY_MIN,
                    day_max=DAY_MAX,
                    folder_name='pv_data',
                    file_name_pv='pv_data_clean.csv',
                    file_name_location='location_data_clean.csv')

    x, y = loader.get_time_series()
    y_in = y[:, IDX]

    x_train, y_train, x_test, y_test = dl.train_test_split(time, y_in, n_hours=3)

    idx = np.random.choice(x_train.shape[0], 350, replace=False)
    x_inducing = x_train[idx]
    y_inducing = y_train[idx]

    # TODO add argparse
    types = ['stochastic']
    names = ['cholesky', 'mean_field']
    num_inducing_points = [x_inducing.size(0), x_train.size(0)]
    days = np.arange(0, 200, 10)
    models = ['BetaSVI', 'GaussianSVI', 'GaussianExact']
