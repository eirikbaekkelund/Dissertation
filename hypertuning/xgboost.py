import optuna
import numpy as np
from models import fit_xgboost
from data import SystemLoader
from metrics import mean_absolute_error

class HyperOptXGBoost:
    def __init__(self, loader : SystemLoader, n_systems : int):
        self.loader = loader
        self.n_systems = n_systems
    
    def sample_params(self, trial):
        """ 
        Parameter sampler for the XGBoost model.
        """
        params = {
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "max_leaves": trial.suggest_int("max_leaves", 1, 10),
            "n_estimators": trial.suggest_int("n_estimators", 1, 500, step=50),
        }
        return params
    
    def objective(self, trial):
        """
        Objective function for the XGBoost model.
        """
        params = self.sample_params(trial)
        losses = []
        break_training = False
        
        for _ in self.loader:
            for i in range(self.n_systems):
                x_tr, y_tr, x_te, y_te = self.loader.train_test_split_individual(i)
                
                if x_tr.shape[0] < 100:
                    break_training = True
                    break
                
                y_pred = fit_xgboost(x_tr, y_tr, x_te, **params)
                mae = mean_absolute_error(y_te.numpy(), y_pred)
                mean_mae = np.mean(mae)
                losses.append(mean_mae)
            
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