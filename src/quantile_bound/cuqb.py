"""Botorch implementation of the
Constrained Upper Quantile Bound (CUQB) Algorithm
Congwen Lu, Joel Paulson
ARXIV link
"""
# import packages
import botorch
import gpytorch
import numpy
import torch
import scipy
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.constraints.constraints import Interval
from botorch import fit_gpytorch_model
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.models import SingleTaskGP
from botorch.models import ModelListGP
from cyipopt import minimize_ipopt
from fast_soft_sort.pytorch_ops import soft_sort
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Optional
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform
from torch import Tensor

# train model function
def train_model(X, Y, nu=1.5, noiseless_obs=True):
  # make sure training data has the right dimension
  if Y.ndim == 1:
      Y = Y.unsqueeze(-1)
  if False:
    model = SingleTaskGP(X, Y)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
  else:  
    # outcome transform
    standardize = Standardize(m=Y.shape[-1], batch_shape=Y.shape[:-2])
    outcome_transform = standardize
    # covariance module
    covar_module = ScaleKernel(MaternKernel(nu=nu, ard_num_dims=X.shape[-1]))
    # likelihood
    if noiseless_obs:
      _, aug_batch_shape = SingleTaskGP.get_batch_dimensions(
          train_X=X,
          train_Y=Y,
      )
      likelihood = GaussianLikelihood(
          batch_shape=aug_batch_shape,
          noise_constraint=Interval(lower_bound=1e-4, upper_bound=1e-3),
      )
    else:
      likelihood = None
    # define the model
    model = SingleTaskGP(
      train_X=X,
      train_Y=Y,
      covar_module=covar_module,
      likelihood=likelihood,
      outcome_transform=outcome_transform,
    )
    # call the training procedure
    model.outcome_transform.eval()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
  fit_gpytorch_model(mll)
  # put in eval mode
  model.eval()
  # return the model
  return model

def jacobian(outputs, inputs, create_graph=False):
    """Computes the jacobian of outputs with respect to inputs

    :param outputs: tensor for the output of some function
    :param inputs: tensor for the input of some function (probably a vector)
    :param create_graph: set True for the resulting jacobian to be differentible
    :returns: a tensor of size (outputs.size() + inputs.size()) containing the
        jacobian of outputs with respect to inputs
    """
    jac = outputs.new_zeros(outputs.size() + inputs.size()
                            ).view((-1,) + inputs.size())
    for i, out in enumerate(outputs.view(-1)):
        col_i = torch.autograd.grad(out, inputs, retain_graph=True,
                              create_graph=create_graph, allow_unused=True)[0]
        if col_i is None:
            # this element of output doesn't depend on the inputs, so leave gradient 0
            continue
        else:
            jac[i] = col_i

    if create_graph:
        jac.requires_grad_()

    return jac.view(outputs.size() + inputs.size())

class CompositeQuantile(MCAcquisitionFunction):
    def __init__(
        self,
        h: Model,
        A: torch.tensor,
        g = None,
        alpha = 0.95,
        num_samples = 50,
        regularization_strength = 0.1,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super(MCAcquisitionFunction, self).__init__(model=h)
        self.A = A
        self.num_samples = num_samples
        self.g = g
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_samples]))
        self.sampler = sampler
        self.regularization_strength = regularization_strength
        self.register_buffer("alpha", torch.as_tensor(alpha))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate qQuantile on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            Tensor: A `(b)`-dim Tensor of Quantile values at the
                given design points `X`.
        """
        A = self.A.to(torch.float32)
        X = X.to(torch.float32)
        Z = torch.bmm(X, A.T.repeat((X.shape[0],1,1)))
        posterior = self.model.posterior(Z)
        Y = self.sampler(posterior)  # n x b x q x o  (output samples)
        Y = Y.squeeze(dim=-2)  # n x b x o      (remove q since it is always = 1)
        if self.g is not None:
          X = X.squeeze(dim=1) #  b x d (remove q)
          X = X.repeat((self.num_samples,1,1)) # n x b x d (repeat for number of samples)
          G = self.g(X, Y)
        soft_sort_G = soft_sort(G.T, regularization_strength=self.regularization_strength) # b x n
        soft_sort_G = soft_sort_G.T # n x b
        quantile = soft_sort_G[int(self.alpha*self.num_samples)-1,:].squeeze() # b
        return quantile
