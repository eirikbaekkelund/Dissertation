import optuna
import numpy as np
from data import SystemLoader
from models import LSTM
from metrics import mean_absolute_error

class HyperOptLSTM:
    def __init__(self, loader : SystemLoader, n_systems : int):
        self.loader = loader
        self.n_systems = n_systems
    
    def sample_params(self, trial):
        """ 
        Parameter sampler for the LSTM model.
        """
        params = {
            "hidden_units": trial.suggest_int("hidden_units", 1, 100, step=5),
            "n_layers": trial.suggest_int("n_layers", 1, 5),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "batch_size": trial.suggest_int("batch_size", 1, 100, step=5),
        }
        return params
    
    def objective(self, trial):
        """
        Objective function for the LSTM model.
        """
        params = self.sample_params(trial)
        n_iter = trial.suggest_int("n_iter", 1, 500, step=50)
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

        losses = []
        break_training = False
        for _ in self.loader:
            for i in range(self.n_systems):
                x_tr, y_tr, x_te, y_te = self.loader.train_test_split_individual(i)
                if x_tr.shape[0] < 100:
                    break_training = True
                    break
                model = LSTM(x_tr, y_tr, **params)
                model.fit(n_iter=n_iter, lr=lr)
                y_pred = model.predict(x_te, y_te, batch_size=params["batch_size"])
                losses.append(mean_absolute_error(y_te, y_pred))
            
            if break_training:
                break

        return np.mean(losses)
    
    def run_study(self, n_trials : int = 100):
        """
        Run the hyperparameter optimization study.
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        return study