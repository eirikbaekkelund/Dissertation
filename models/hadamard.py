import torch
import gpytorch
import wandb
import numpy as np
from gpytorch.variational import (CholeskyVariationalDistribution, 
                                  VariationalStrategy,
                                  IndependentMultitaskVariationalStrategy,
                                  LMCVariationalStrategy)
from torch.utils.data import TensorDataset, DataLoader
from typing import Optional


class HadamardGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, 
        X : torch.Tensor,
        y : torch.Tensor,
        mean_module : gpytorch.means.Mean,
        covar_module : gpytorch.kernels.Kernel,
        likelihood : gpytorch.likelihoods.Likelihood,
        num_tasks : int,
        learn_inducing_locations : bool = True,
        use_LMC : bool = False,
        num_latents : Optional[int] = 1,
        jitter : Optional[float] = None
    ):

        # need to ensure that y_train is in (0, 1)
        if y.max() >= 1:
            y[y >= 1] = 1 - 1e-4
        if y.min() <= 0:
            y[y <= 0] = 1e-4
        
        inducing_points = torch.rand(X.size(0) // (num_tasks + 1), X.size(1))

        if use_LMC:
            assert num_latents > 1, "Must specify num_latents > 1 when using LMC"
            variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0),
            batch_shape=torch.Size([num_latents]),
            jitter=jitter
        )
            variational_strategy = LMCVariationalStrategy(
                VariationalStrategy(
                    self,
                    inducing_points=inducing_points,
                    variational_distribution=variational_distribution,
                    learn_inducing_locations=learn_inducing_locations,
                    jitter_val=jitter
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            jitter_val=jitter

        )
        else:
            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=X.size(0),
                batch_shape=torch.Size([num_tasks]),
                jitter=jitter
            )
            variational_strategy = IndependentMultitaskVariationalStrategy(
                VariationalStrategy(
                    self, 
                    inducing_points=X, 
                    variational_distribution=variational_distribution, 
                    learn_inducing_locations=learn_inducing_locations,
                    jitter_val=jitter
                ),
                num_tasks=num_tasks,
            )

        super().__init__(variational_strategy)
       
        self.X = X
        self.y = y
        self.mean_module = mean_module
        self.covar_module = covar_module
        self.likelihood = likelihood        

    def forward(self, x):
     
        mean = self.mean_module(x)
        covar = self.covar_module(x)
    
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    
    def fit_full(
        self,
        n_iter : int,
        lr : float,
        task_indices : torch.Tensor,
        optimizer : torch.optim.Optimizer,
        elbo : gpytorch.mlls.VariationalELBO,
        verbose : bool = False,
        use_wandb : bool = False
    ):
   
        print_freq = 10
        # TODO natural gradients
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
    
    def fit_minibatch(
        self,
        n_iter : int,
        lr : float,
        task_indices : torch.Tensor,
        optimizer : torch.optim.Optimizer,
        elbo : gpytorch.mlls.VariationalELBO,
        verbose : bool = False,
        use_wandb : bool = False
    ):
       
        print_freq = 10
        # TODO natural gradients
        # create dataloader for minibatching
        
        dataset = TensorDataset(self.X, self.y, task_indices)
        dataloader = DataLoader(dataset, batch_size=500, shuffle=True)
        
        for i in range(n_iter):
            for x_batch, y_batch, task_batch in dataloader:
                optimizer.zero_grad()
                output = self(x_batch, task_indices=task_batch)
                loss = -elbo(output, y_batch, task_indices=task_batch)
                loss.backward()
                optimizer.step()

                if verbose and (i + 1) % print_freq == 0:
                    print(f"Iter {i + 1}/{n_iter} - Loss: {loss.item()}")

                if use_wandb and i % print_freq == 0:
                    pass 
                    # TODO log to wandb
        
    def fit(
        self,
        n_iter : int,
        lr : float,
        task_indices : torch.Tensor,
        verbose : bool = False,
        use_wandb : bool = False,
        minibatch : bool = False
    ):

        self.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elbo = gpytorch.mlls.VariationalELBO(likelihood=self.likelihood,
                                             model=self,
                                             num_data=self.y.size(0))
        
        if minibatch:
            self.fit_minibatch(n_iter, lr, task_indices, optimizer, elbo, verbose, use_wandb)
        else:
            self.fit_full(n_iter, lr, task_indices, optimizer, elbo, verbose, use_wandb)

    
    def predict(self, x_test, task_indices):
        self.eval()
        self.likelihood.eval()
        task_indices = task_indices.to(dtype=torch.long, device=x_test.device)
        with torch.no_grad():
            preds = self.likelihood(self(x_test, task_indices=task_indices), task_indices=task_indices)
            samples = preds.sample(sample_shape=torch.Size([100]))
            samples = samples.mean(dim=0)
     
        # TODO finish this
            
