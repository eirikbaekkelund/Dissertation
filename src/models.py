import torch
import gpytorch

# TODO make Beta likelihood work (possibly inherit from ExactGP)

class ExactGPModel(gpytorch.models.ExactGP):
    """ 
    Class for exact GP model.

    Args:
        train_x (torch.Tensor): training data
        train_y (torch.Tensor): training labels
        likelihood (gpytorch.likelihoods.Likelihood): likelihood
        mean_module (gpytorch.means.Mean): mean module
        covar_module (gpytorch.kernels.Kernel): covariance module
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
        self.train_x = train_x
        self.train_y = train_y
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

    def _train(self, n_iter, lr, optim):
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

    def predict(self, x_test):
        """ 
        Make prediction on test data

        Args:
            x_test (torch.Tensor): test data
        
        Returns:
            preds_test (torch.Tensor): predicted mean test
        """
        self.eval()
        self.likelihood.eval()
        
        with torch.no_grad():
            preds_test = self.likelihood(self(x_test))
            preds_train = self.likelihood(self(self.train_x))
        
        return preds_test, preds_train

