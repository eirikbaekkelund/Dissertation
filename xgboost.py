import xgboost as xgb
from xgboost import XGBRegressor
from gpytorch.distributions import MultivariateNormal

class XGB(XGBRegressor):
    def __init__(self, params):
        self.params = params
        super().__init__(**params)
    
    # simulate for probabilistic forecasting
    def simulate(self, X, n_samples=1000):
        """
        Approximate a normal distribution for the target variable
        using the XGBoost model and the given input data. The 
        distribution is then sampled from to generate probabilistic
        forecasts.
        """
        # get the mean and standard deviation of the target variable
        # using the XGBoost model
        # simulate N sample predictions from the distribution
        samples = self.predict(X, ntree_limit=self.best_ntree_limit,
        
        # create a normal distribution
        dist = MultivariateNormal(mean, std)
