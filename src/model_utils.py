import torch
import gpytorch

class ExactGPModel(gpytorch.models.ExactGP):
    """ 
    Class for exact GP model.

    Args:
        train_x (torch.Tensor): training data
        train_y (torch.Tensor): training labels
        likelihood (gpytorch.likelihoods): likelihood
        mean_module (gpytorch.means): mean module
        covar_module (gpytorch.kernels): covariance module
    """
    def __init__(self, 
                 train_x, 
                 train_y, 
                 likelihood,
                 mean_module,
                 covar_module):
        
        super(ExactGPModel, self).__init__(train_x, 
                                           train_y, 
                                           likelihood)
        
        self.mean_module = mean_module
        self.covar_module = covar_module
    
    def forward(self, x):
        """ 
        Model prediction of the GP model

        Args:
            x (torch.Tensor): input data
        
        Returns:
            gpytorch.distributions.MultivariateNormal: GP model 
                                                       f ~ GP( m(x), k(x, x)) 
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def train(self, n_iter, lr, optim):
        """
        Train the GP model

        Args:
            n_iter (int): number of iterations
            lr (float): learning rate
            optim (str): optimizer
        """
        
        self.train()
        self.likelihood.train()
        
        if optim == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optim == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        else:
            raise ValueError('optim must be Adam or SGD')
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        for i in range(n_iter):
            optimizer.zero_grad()
            output = self(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
<<<<<<< HEAD

=======
>>>>>>> 0201744 (exact gp model)
