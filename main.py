import torch 
import numpy as np
from data import PVDataGenerator
from data.utils import *
from pv_plot import plot_gp
from models import ApproximateGPBaseModel
from likelihoods.beta import BetaLikelihood_MeanParametrization
from kernels import Kernel
from gpytorch.means import ZeroMean
from gpytorch.constraints import Interval, Positive
from gpytorch.metrics import negative_log_predictive_density as nlpd
from gpytorch.utils.errors import NotPSDError

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

if __name__ == "__main__":
    DAY_INIT = 20
    DAY_MIN = 8
    DAY_MAX = 16
    N_DAYS = 5
    MINUTE_INTERVAL = 5
    DAILY_DATA_POINTS = (DAY_MAX - DAY_MIN) * 60 / MINUTE_INTERVAL
    N_HOURS_PRED = 8
    N_SYSTEMS = 300
    RADIUS = 0.35
    COORDS = (55, -1.5)
    POLY_COORDS = ((50, -6), (50.5, 1.9), (57.6, -5.5), (58, 1.9))
    JITTER = 1e-3

    generator = generator = PVDataGenerator(n_days=N_DAYS,
                    day_init=DAY_INIT,
                    n_systems=N_SYSTEMS,
                    radius=RADIUS,
                    coords=POLY_COORDS,
                    minute_interval=MINUTE_INTERVAL,
                    day_min=DAY_MIN,
                    day_max=DAY_MAX,
                    folder_name='pv_data',
                    file_name_pv='pv_data_clean.csv',
                    file_name_location='location_data_clean.csv',
                    distance_method='poly',
                    drop_nan=True)
    
    # randomly select 10 systems
    X, y = generator.get_time_series()
    idx = np.random.choice(y.size(-1) - 1, size=10, replace=False)
    y = y[:, idx]

    # configure model
    mean = ZeroMean()
    kernel = Kernel()

    x_train, y_train, x_test, y_test = train_test_split(X, y, hour=13)

    matern_base = kernel.get_matern(lengthscale_constraint=Positive())
    matern_quasi = kernel.get_matern(lengthscale_constraint=Interval(0.3, 1000.0))
    periodic1 = kernel.get_periodic(lengthscale_constraint= Positive())
    periodic2 = kernel.get_periodic(lengthscale_constraint= Interval(0.1, 1000.0))

    quasi_periodic = kernel.get_quasi_periodic(matern_base=matern_base, 
                                            matern_quasi=matern_quasi,
                                            periodic1=periodic1,
                                            periodic2=None)
    config = {  'type': 'stochastic',
            'name': 'mean_field',
            'num_inducing_points': x_train.size(0)
            }
    
    inputs ={
        'X' : x_train,
        'config': config,
        'jitter': JITTER,
        'likelihood': BetaLikelihood_MeanParametrization(),
        'mean_module': mean,
        'covar_module': quasi_periodic,
        'learn_inducing_locations': False
    }

    for idx in range(y_train.size(-1)):
        inputs['y'] = y_train[:,idx]
        
        model_beta = ApproximateGPBaseModel(**inputs)
        try:
            model_beta.fit(n_iter=400, lr=0.1,  verbose=True, use_wandb=True)
            plot_gp(model_beta, x_train, x_test, y_train[:,idx], y_test[:,idx], 'all')
        
        except NotPSDError as e:
            print(e, 'skipping...')
