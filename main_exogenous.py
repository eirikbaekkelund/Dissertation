
from models import (ApproximateGPBaseModel,
                    HadamardGPModel,
                    LSTM, 
                    fit_bayesian_ridge,
                    fit_xgboost)
from likelihoods import BetaLikelihood_MeanParametrization, HadamardBetaLikelihood
from data import SystemLoader, PVWeatherGenerator
from data.utils import train_test_split


if __name__ == "__main__":
    # data parameters (Data is from Manchester Area)
    DAY_INIT = 0 # start of data is 01-01-2018
    N_DAYS = 365 # data until 01-01-2019
    DAY_MIN = 8 # starting hour of all our data (per day)
    DAY_MAX = 16 # ending hour of all our data (per day)
    MINUTE_INTERVAL = 5 # the minute interval of our data
    DAILY_DATA_POINTS = (DAY_MAX - DAY_MIN) * 60 // MINUTE_INTERVAL
    N_DAYS_FOLD = 8
    N_SYSTEMS = 10
    CIRCLE_COORDS = (53.5, -3)
    RADIUS = 0.4
    MODELS = ['HadamardGP', 'LSTM', 'BayesianRidge', 'XGBoost', 'SimpleGP']
    
    # generator takes data from the entire data base and filters out what we specify 
    generator = PVWeatherGenerator(
        coords=CIRCLE_COORDS,
        radius=RADIUS,
        day_init=DAY_INIT,
        n_systems=N_SYSTEMS,
        n_days=N_DAYS,
        minute_interval=MINUTE_INTERVAL,
    )
    df = generator.df
     # columns to use for the data (latitude and longitude will be dropped - no statistical correlation)
    X_COLS = ['global_rad:W', 'diffuse_rad:W',
                'effective_cloud_cover:octas',
                'relative_humidity_2m:p', 't_2m:C',
                'wind_speed_10m:ms', 'latitude', 'longitude']
    TARGET_COL = 'PV'
    
    X, y = df[X_COLS], df[TARGET_COL]

    INTERVAL = int(DAILY_DATA_POINTS * N_DAYS_FOLD) # number of data points per system in each fold
    # create loader that will iterate over the data
    loader = SystemLoader(X, y, train_interval=INTERVAL)

    # X is the data, Y is the target, T is the task indices as an array for Hadamard GP
    print(f"start index: {loader.start} | end index: {loader.end}")
    for X_tr, Y_tr, X_te, Y_te, T_tr, T_te in loader:
        # concatenated systems are in X, Y, T
        # which are split into train and test data
        print(X_tr.shape, Y_tr.shape, X_te.shape, Y_te.shape, T_tr.shape, T_te.shape)
        # if you want to do it by individual systems:
        for i in range(loader.n_systems):
            x_tr, y_tr, x_te, y_te = loader.train_test_split_individual(i)
            break
        # updated slicing
        print(f"start index: {loader.start} | end index: {loader.end}")
        break
        
        # TODO, remember to add temporal component to X, could be done in the loader
