import torch
import gpytorch
from gpytorch.variational import IndependentMultitaskVariationalStrategy

# TODO add choice between independent and LMC
class HadamardGP(gpytorch.models.ApproximateGP):
    def __init__(self, 
                X : torch.Tensor,
                y : torch.Tensor,
                mean_module : gpytorch.means.Mean,
                covar_module : gpytorch.kernels.Kernel,
                likelihood : gpytorch.likelihoods.Likelihood,
                num_tasks : int):

        # need to ensure that y_train is in (0, 1)
        if y.max() >= 1:
            y[y >= 1] = 1 - 1e-4
        if y.min() <= 0:
            y[y <= 0] = 1e-4

        # We have to mark the variational distribution as batch to lean 
        # separate variational parameters for each task
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(
            X.size(0), batch_shape=torch.Size([num_tasks])
        )
        # TODO Consider LMC as well
        variational_strategy = IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, X, variational_distribution, learn_inducing_locations=False
            ),
            num_tasks=num_tasks,
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.X = X
        self.y = y
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.likelihood = likelihood
        

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
    
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def fit(self,
            n_iter : int,
            lr : float,
            task_indices : torch.Tensor,
            verbose : bool = False,
            use_wandb : bool = False):
        self.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elbo = gpytorch.mlls.VariationalELBO(likelihood=self.likelihood, 
                                             model=self, 
                                             num_data=self.y.size(0))

        print_freq = 10

        for i in range(n_iter):
            optimizer.zero_grad()
            output = self(self.X, task_indices=task_indices)
            loss = -elbo(output, self.y, task_indices=task_indices)
            loss.backward()
            optimizer.step()

            if verbose and (i + 1) % print_freq == 0:
                print(f"Iter {i + 1}/{n_iter} - Loss: {loss.item()}")

            if use_wandb and i % print_freq == 0:
                pass 
                # TODO log to wandb