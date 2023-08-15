import numpy as np
from data import PVDataGenerator, PVDataLoader
from data.utils import (cross_val_fold, 
                        train_test_split_fold)
from hypertuning import MultitaskBetaQPGP

if __name__ == '__main__':
    # data parameters
    DAY_INIT = 0
    DAY_MIN = 8
    DAY_MAX = 16
    N_DAYS = 365
    N_DAYS_TRAIN = 5
    MINUTE_INTERVAL = 5
    DAILY_DATA_POINTS = (DAY_MAX - DAY_MIN) * 60 / MINUTE_INTERVAL
    N_HOURS_PRED = 2
    N_SYSTEMS = 15
    RADIUS = 0.35
    CIRCLE_COORDS = (55, -1.5)

    for season in ['winter', 'spring', 'summer', 'fall']:
        generator = PVDataGenerator(n_days=N_DAYS,
                    day_init=DAY_INIT,
                    n_systems=N_SYSTEMS,
                    coords=CIRCLE_COORDS,
                    minute_interval=MINUTE_INTERVAL,
                    day_min=DAY_MIN,
                    day_max=DAY_MAX,
                    folder_name='pv_data',
                    file_name_pv='pv_data_clean.csv',
                    file_name_location='location_data_clean.csv',
                    distance_method='circle',
                    season='spring',
                    drop_nan=False)

        X, y = generator.get_time_series()
        # run hypertuning on ~a month of data
        y = y[:int(DAILY_DATA_POINTS*30), :]
        X = X[:int(DAILY_DATA_POINTS*30)]
        
        x_list, y_list = cross_val_fold(X, y, N_DAYS_TRAIN, DAILY_DATA_POINTS)
        x_train, y_train, x_test, y_test =  train_test_split_fold(x_list, y_list, N_HOURS_PRED, MINUTE_INTERVAL, DAY_MIN, DAY_MAX)
        
        train_loader = PVDataLoader(x_train, y_train)
        test_loader = PVDataLoader(x_test, y_test)
        
        model = MultitaskBetaQPGP(train_loader=train_loader,
                                  test_loader=test_loader)
        model.run_study(n_trials=30, direction='minimize')
        model.save_best_params(f'multitask_beta_qpgp_{season}')

