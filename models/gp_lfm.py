import torch
import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import VariationalStrategy
from models.variational import VariationalBase

class ApproximateGP(gpytorch.models.ApproximateGP):
    def __init__(self, dataset, mean, covar, likelihood, config):
        
        config['num_inducing_points'] = dataset[0][0].shape[0]
        variational_dist = VariationalBase(config).variational_distribution
        
        variational_strategy = VariationalStrategy(
            self,
            variational_distribution=variational_dist,
            inducing_points=dataset[0][0],
            learn_inducing_locations=True)
        
        super().__init__(variational_strategy)
        
        self.dataset = dataset
        self.mean_module = mean
        self.covar_module = covar
        self.likelihood = likelihood
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
    def predict(self, x):
        with torch.no_grad():
            dist = self.likelihood(self(x))
        return dist
    
    def fit(self, n_iter, lr, verbose=False):

        optim = torch.optim.Adam(self.parameters(), lr=lr)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self, num_data=len(self.dataset[0][0]))

        early_stop = 0
        losses = []

        for i in range(n_iter):
            optim.zero_grad()
            output = self(self.dataset[0][0])
            loss = -mll(output, self.dataset[0][1])
            loss.backward()
            optim.step()
            
            if verbose and i + 1  % 10 == 0:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
            
            losses.append(loss.item())

            if i > 0 and abs(losses[-1] - losses[-2]) < 1e-5:
                early_stop += 1
                if early_stop > 15:
                    print('Early stopping of GP')
                    break
        

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, mean, covar, likelihood):
        super().__init__(x_train, y_train, likelihood)
        self.mean_module = mean
        self.covar_module = covar
        self.likelihood = likelihood
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
    def predict(self, x):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad():
            return self.likelihood(self(x))
    
    def fit(self,n_iter, lr, verbose, print_freq=10):
        self.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        early_stop = 0
        losses = []
        for i in range(n_iter):
            optimizer.zero_grad()
            output = self(self.train_inputs[0])
            loss = -mll(output, self.train_targets)
            loss.backward()
            optimizer.step()
            if verbose and i % print_freq == 0:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
            
            losses.append(loss.item())

            if i > 0 and abs(losses[-1] - losses[-2]) < 1e-5:
                early_stop += 1
                if early_stop > 15:
                    print('Early stopping of GP')
                    break
            
