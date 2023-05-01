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
        num_samples = 100,
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

class BlackBoxFunction:
    def __init__(self, f, ninput, beta=-2.0, input_data=None, output_data=None, acq_type = None):
      self.f = f
      self.ninput = ninput
      self.beta = beta
      self.input_data = input_data
      self.output_data = output_data
      self.model = None
      self.acq_type = acq_type

    def train_model(self):
      self.model = train_model(self.input_data, self.output_data, nu=1.5, noiseless_obs=True)

    def add_data(self, new_input, new_output):
      if len(new_output.shape) == 1:
        new_output = new_output.reshape((-1,1))
      if self.input_data is None:
        self.input_data = new_input
        self.output_data = new_output
      else:
        self.input_data = torch.vstack((self.input_data, new_input))
        self.output_data = torch.cat((self.output_data, new_output))       

    def update_beta(self, beta):
      # need to overwrite previous beta 
      self.beta = beta

    def eval_cb_torch(self, input):
      # need to make sure that input has dimension d x b (number of inputs by batch)
      if len(input.shape) == 1:
        input = input.unsqueeze(-1)
      posterior = self.model(input.T)
      mean = posterior.mean
      std = torch.sqrt(posterior.variance)
      beta = torch.tensor(self.beta, dtype=torch.float64)
      return mean + beta*std

class WhiteBoxFunction:
    def __init__(self, f, ninput):
      self.f = f
      self.ninput = ninput

    def eval_torch(self, input):
      # need to make sure that input has dimension d x b (number of inputs by batch)
      if len(input.shape) == 1:
        input = input.unsqueeze(-1)
      output = self.f(input.T)
      return output

class GreyBoxFunction:
    def __init__(self, g, h, A, input_data=None, output_data=None, alpha=0.05, num_samples=100, reg_strength=0.1):
      self.g = g
      self.h = h
      self.A = A
      self.input_data = input_data
      self.output_data = output_data
      self.model_list = None
      self.model = None
      self.alpha = alpha
      self.num_samples = num_samples
      self.reg_strength = reg_strength
      self.ninput = A.shape[1]

    def add_data(self, new_input, new_output):
      if len(new_output.shape) == 1:
        new_output = new_output.reshape((-1,1))
      if self.input_data is None:
        self.input_data = new_input
        self.output_data = new_output
      else:
        self.input_data = torch.vstack((self.input_data, new_input))
        self.output_data = torch.cat((self.output_data, new_output))

    def train_model(self):
      self.model_list = []
      noutput = self.output_data.shape[1]
      for i in range(noutput):
        model_i = train_model(self.input_data, self.output_data[:,i], nu=1.5, noiseless_obs=True)
        self.model_list += [model_i]
      self.model = ModelListGP(*self.model_list)
      self.create_qb_torch() # always need to create a new torch quantile function when model is trained

    def update_alpha(self, alpha):
      # need to overwrite previous alpha 
      self.alpha = alpha
      self.create_qb_torch()

    # need to create a torch version of the composite quantile
    def create_qb_torch(self):
      self.qb = CompositeQuantile(self.model, A=self.A, alpha=self.alpha, num_samples=self.num_samples, g=self.g, regularization_strength=self.reg_strength)

    # do evaluations using torch 
    def eval_qb_torch(self, input):
      x = input
      if len(x.shape) == 1: # need to make sure that we can interface with acqusition class (always takes in b x q x d shape)
        x = x.view(1,1,x.shape[0])
      elif len(x.shape) == 2:
        x = x.T.unsqueeze(1)
      acq = self.qb(x)
      return acq
  
class Problem:
    def __init__(self, fun_list, fun_type, h, A, xL, xU, alpha_list, f_opt=None, x_opt=None):
      self.fun_list = fun_list
      self.fun_type = fun_type
      self.h = h
      self.A = A
      self.xL = xL
      self.xU = xU
      self.alpha_list = alpha_list
      self.f_opt = f_opt
      self.x_opt = x_opt
      
class Configuration:
  def __init__(self, problem, acq_f = None, Ninit=5, NFE=5, x_data=None, z_data=None, h_data=None, fun_data=None, n_samp_quantile=100, M=1e6, eps=1e-3, opt_method="ipopt", raw_samples=1000, num_multi_starts=1, seed=0):
    self.problem = problem
    self.acq_f = acq_f
    self.acq_func = None
    self.orignal_fun_list = problem.fun_list
    self.fun_type = problem.fun_type
    self.h = problem.h # composite black-box function
    self.A = problem.A
    self.xL = problem.xL
    self.xU = problem.xU
    self.Ninit = Ninit
    self.NFE = NFE
    self.x_data = x_data
    self.z_data = z_data
    self.h_data = h_data
    self.fun_data = fun_data
    self.alpha_list = problem.alpha_list
    self.n_samp_quantile = n_samp_quantile
    self.nx = problem.A.shape[1]
    self.fun_list = self.convert_fun_to_classes()
    self.x_opt_list = None
    self.f_opt_list = None
    self.g_opt_list = None
    self.M = M # value used in penalty
    self.eps = eps # value used to cutoff constraints
    self.opt_method = opt_method
    self.raw_samples = raw_samples
    self.num_multi_starts = num_multi_starts
    self.seed = seed

  def convert_fun_to_classes(self):
    flist = self.orignal_fun_list
    ftype = self.fun_type
    converted_fun_list = []
    bb_list = []
    for (i,f) in enumerate(flist):
      if ftype[i] == "wb":
        converted_fun_list += [WhiteBoxFunction(f, self.nx)]

      elif ftype[i] == "bb":
        converted_fun_list += [BlackBoxFunction(f, self.nx, beta=norm.ppf(self.alpha_list[i]))]
        
      elif ftype[i] == "gb":
        converted_fun_list += [GreyBoxFunction(f, self.h, self.A, alpha=self.alpha_list[i], num_samples=self.n_samp_quantile)]

    return converted_fun_list

  def eval_new_samples(self, new_x):
    # start by evaluating the composite black-box stored locally
    if self.A is None:
      new_z = None
    else:
      new_x = new_x.to(torch.float64)
      A = self.A.to(torch.float64)
      new_z = torch.matmul(new_x, A.T)
      new_z = new_z.to(torch.float64)
    if self.h is None:
      new_h = None
    else:
      new_h = self.h(new_z)

    # next go through all functions in list to evaluate them
    fun_list = self.fun_list
    ftype = self.fun_type
    new_fun = torch.zeros((new_x.shape[0],len(fun_list)), dtype=torch.float64)
    for (i,fun) in enumerate(fun_list):
      if ftype[i] == "wb":
        new_fun[:,i] = fun.f(new_x)
      elif ftype[i] == "bb":
        #print(fun.f(new_x).shape)
        if fun.f(new_x).shape[0] > 1:
            new_fun[:,i] = fun.f(new_x).T
        else:
            new_fun[:,i] = fun.f(new_x)
      elif ftype[i] == "gb":
        new_fun[:,i] = fun.g(new_x, new_h)

    # return the relevant evaluations
    return new_z, new_h, new_fun

  def add_new_samples(self, new_x, new_z, new_h, new_fun):
    # store new samples locally
    if self.x_data is None:
      self.x_data = new_x
    else:
      self.x_data = torch.vstack((self.x_data, new_x))
    if self.z_data is None:
      self.z_data = new_z
    else:
      self.z_data = torch.vstack((self.z_data, new_z))
    if self.h_data is None:
      self.h_data = new_h
    else:
      self.h_data = torch.vstack((self.h_data, new_h))
    if self.fun_data is None:
      self.fun_data = new_fun
    else:
      self.fun_data = torch.vstack((self.fun_data, new_fun))

    # be sure to update the data in the black- and grey-box functions
    fun_list = self.fun_list
    ftype = self.fun_type
    for (i,fun) in enumerate(fun_list):
      if ftype[i] == "bb":
        fun.add_data(new_x, new_fun[:,i])
      elif ftype[i] == "gb":
        fun.add_data(new_z, new_h)

  def train_models(self):
    # need to loop over functions and train all black- and grey-box models
    fun_list = self.fun_list
    ftype = self.fun_type
    for (i,fun) in enumerate(fun_list):
      if ftype[i] == "bb" or ftype[i] == "gb":
        fun.train_model()

  def get_batch_initial_conditions(self):
    # generate some random values (make sure d x b)
    Xraw = self.xL + (self.xU - self.xL) * torch.rand(self.raw_samples, self.nx)
    Xraw = Xraw.T

    # loop over functions and evaluate objective + penalized constraints
    Yraw = torch.zeros(self.raw_samples)
    fun_list = self.fun_list
    ftype = self.fun_type
    if self.acq_f == "EI":
        self.create_EI()
        #print(Xraw.float().dtype)
        Yraw = self.eval_ei_torch(Xraw.double())
    else:
        for (i,fun) in enumerate(fun_list):
          if ftype[i] == "wb":
            fun_eval = fun.eval_torch(Xraw)
          elif ftype[i] == "bb":
            fun_eval = fun.eval_cb_torch(Xraw.double())
          elif ftype[i] == "gb":
            fun_eval = fun.eval_qb_torch(Xraw)
          if i == 0:  # objective
            Yraw += fun_eval
          else:       # constraint 
            Yraw += self.M * torch.max( fun_eval.reshape((-1,1)), torch.zeros((self.raw_samples, 1)) ).squeeze()
    
    # use BOTorch built-in functions to determine multi-start locations
    batch_initial_conditions = botorch.optim.initializers.initialize_q_batch(Xraw.T, Yraw, self.num_multi_starts)
    return batch_initial_conditions
  
  def create_EI(self):
    # create EI model, need: list of GP models
    # first, calcualte incumbent
    n_iter = len(self.fun_data[:,0])

    inc_x, inc_f, inc_g = self.get_current_best_point(iter=n_iter)
    #min_incumbent = torch.min(self.fun_data[:,0].unsqueeze(dim=1) + 1e4*(torch.sum(self.fun_data[:,1:], dim=1).unsqueeze(dim=1)))
    min_incumbent = inc_f

    # loop over and calculate constraints:
    if len(fun_list) == 1:
        ng = 0
    else:
        ng = len(fun_list[1:])
        constraints = {1: (None, 0.0)}
        if len(fun_list[1:]) > 1:
            for i in range(ng-1):
                constraints.update({i+2: (None, 0.0)})
    
    # create EI (for minimizing)
    if ng == 0:
        acq_func = botorch.acquisition.analytic.ExpectedImprovement(self.fun_list[0].model, min_incumbent, maximize=False)
    else:
        gp_list = []
        for (i,fun) in enumerate(self.fun_list):
            gp_list.append(fun.model)
            
        obj_and_cons = ModelListGP(*gp_list)
        #x_opt = torch.Tensor([[0.1951, 0.4047]])
        acq_func = botorch.acquisition.analytic.ConstrainedExpectedImprovement(obj_and_cons, min_incumbent, 0, constraints, maximize=False)
    #print(obj_and_cons.posterior(x_opt).mean)
    self.acq_func = acq_func
    return self.acq_func
  
  def eval_ei_torch(self, input):
      x = input

      if len(x.shape) == 1: # need to make sure that we can interface with acqusition class (always takes in b x q x d shape)
        x = x.view(1,1,x.shape[0])
      elif len(x.shape) == 2:
        x = x.T.unsqueeze(1)
        
      acq_ei = -1*(self.acq_func(x))
      return acq_ei
    
  def optimize_acq(self):
    raw_samples = self.raw_samples
    method = self.opt_method
    if method == "random":
      # generate some random values (make sure d x b)
      Xraw = self.xL + (self.xU - self.xL) * torch.rand(raw_samples, self.nx)
      Xraw = Xraw.T
      # loop over functions and evaluate objective + penalized constraints
      Yraw = torch.zeros(raw_samples)
      fun_list = self.fun_list
      ftype = self.fun_type
      for (i,fun) in enumerate(fun_list):
        if ftype[i] == "wb":
          fun_eval = fun.eval_torch(Xraw)
        elif ftype[i] == "bb":
          fun_eval = fun.eval_cb_torch(Xraw)
        elif ftype[i] == "gb":
          fun_eval = fun.eval_qb_torch(Xraw)        
        if i == 0:  # objective
          Yraw += fun_eval
        else:       # constraint 
          Yraw += self.M * torch.max( fun_eval.reshape((-1,1)), torch.zeros((raw_samples, 1)) ).squeeze()

      # find the best solution
      min_index = torch.argmin(Yraw)
      x_opt_final = Xraw[:,min_index]

    elif method == "ipopt" or "scipy":
      # shorthand names
      fun_list = self.fun_list
      ftype = self.fun_type

      # executing the solver
      if len(fun_list) == 1:
        ng = 0
      else:
        ng = len(fun_list[1:])

      # need objective function and its gradient
      def objective(x, return_grad=False):
        if return_grad is False:
          x = torch.tensor(x, dtype=torch.float64)
          if ftype[0] == "wb":
            fun_eval = fun_list[0].eval_torch(x)
          elif ftype[0] == "bb":
            if self.acq_f == "EI":
                self.create_EI()
                fun_eval = self.eval_ei_torch(x)
            else:
                fun_eval = fun_list[0].eval_cb_torch(x)
          elif ftype[0] == "gb":
            fun_eval = fun_list[0].eval_qb_torch(x)
          fun_eval = fun_eval.detach()
          return fun_eval.numpy()
        else:
          x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
          if ftype[0] == "wb":
            fun_eval = fun_list[0].eval_torch(x)
          elif ftype[0] == "bb":
            if self.acq_f == "EI":
                self.create_EI()
                fun_eval = self.eval_ei_torch(x)
            else:
                fun_eval = fun_list[0].eval_cb_torch(x)
          elif ftype[0] == "gb":
            fun_eval = fun_list[0].eval_qb_torch(x)
          fun_eval.backward()
          return x.grad.detach().numpy()

      def objective_grad(x):
        return objective(x, return_grad=True)

      if ng > 0:    
        def cons_ineq(x, return_jac=False): # NOTE: need -1 in the evaluation to convert from g(x) <= 0 to -g(x) >= 0
          if return_jac is False:
            x = torch.tensor(x, dtype=torch.float64)
            cons_eval = None
            for (i,fun) in enumerate(fun_list[1:]):
              if ftype[i+1] == "wb":
                fun_eval = -fun.eval_torch(x)
              elif ftype[i+1] == "bb":
                fun_eval = -fun.eval_cb_torch(x)
              elif ftype[i+1] == "gb":
                fun_eval = -fun.eval_qb_torch(x)
              fun_eval = fun_eval.detach()
              if cons_eval is None:
                cons_eval = fun_eval
              else:
                cons_eval = torch.vstack((cons_eval,fun_eval))
            return cons_eval.squeeze().numpy()
          else:            
            x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
            cons_eval = None
            for (i,fun) in enumerate(fun_list[1:]):
              if ftype[i+1] == "wb":
                fun_eval = -fun.eval_torch(x)
              elif ftype[i+1] == "bb":
                fun_eval = -fun.eval_cb_torch(x)
              elif ftype[i+1] == "gb":
                fun_eval = -fun.eval_qb_torch(x)
              if cons_eval is None:
                cons_eval = fun_eval
              else:
                cons_eval = torch.vstack((cons_eval,fun_eval))
            output = cons_eval.squeeze()
            return jacobian(output, x)

        def cons_ineq_jac(x):
          return cons_ineq(x, return_jac=True)

        # constraints
        cons = [ {'type':'ineq', 'fun':cons_ineq, 'jac':cons_ineq_jac} ]

      else:
        cons = ()

      # first get a set of initial conditions
      batch_initial_conditions = self.get_batch_initial_conditions()

      # variable bounds
      bnds = [(self.xL[i].numpy(), self.xU[i].numpy()) for i in range(self.xL.shape[0])]
    
      x_opt_list = None
      f_opt_list = None
      g_opt_list = None
      success_list = None
      for x0 in batch_initial_conditions:  
        x0 = x0.detach().numpy()
        if method == "ipopt":
          if self.acq_f == 'EI':
              cons = ()
              res = minimize_ipopt(objective, jac=objective_grad, x0=x0, bounds=bnds, constraints=cons, tol=1e-4, options={'print_level': 0, 'maxiter':300})
          else:
              res = minimize_ipopt(objective, jac=objective_grad, x0=x0, bounds=bnds, constraints=cons, tol=1e-4, options={'print_level': 0, 'maxiter':300})
        elif method == "scipy":
          if ng > 0:
            scipy_opt_method = "SLSQP"
            res = minimize(objective, jac=objective_grad, x0=x0, bounds=bnds, constraints=cons, method=scipy_opt_method, options={'ftol':1e-4, 'maxiter':500})
          else:
            scipy_opt_method = "L-BFGS-B"
            res = minimize(objective, jac=objective_grad, x0=x0, bounds=bnds, method=scipy_opt_method, options={'ftol':1e-4, 'maxiter':500})
        x_opt = res.x
        f_opt = res.fun
        if method == "ipopt":
          g_opt = -res.info['g']
        elif method == "scipy":
          if ng > 0:
            g_opt = -cons_ineq(x_opt)
          else:
            g_opt = None
          res.success = int(res.success)
        success_opt = res.success
        x_opt = torch.tensor(x_opt).reshape((1,self.nx))
        f_opt = torch.tensor(f_opt).reshape((1,1))
        if ng > 0 and self.acq_f is None:
          g_opt = torch.tensor(g_opt).reshape((1,ng))
        success_opt = torch.tensor(success_opt).reshape((1,1))
        if x_opt_list is None:
          x_opt_list = x_opt
          f_opt_list = f_opt
          g_opt_list = g_opt
          success_opt_list = success_opt
        else:
          x_opt_list = torch.vstack((x_opt_list, x_opt))
          f_opt_list = torch.vstack((f_opt_list, f_opt))
          if ng > 0 and self.acq_f is None:
            g_opt_list = torch.vstack((g_opt_list, g_opt))
          else:
            g_opt_list = None
          success_opt_list = torch.vstack((success_opt_list, success_opt))

      # store list for future reference
      self.x_opt_list = x_opt_list
      self.f_opt_list = f_opt_list
      self.g_opt_list = g_opt_list
      self.success_opt_list = success_opt_list

      # find the best solution out of the multi-start
      if ng > 0 and self.acq_f is None:
        min_index = torch.argmin(f_opt_list.squeeze() + self.M*(torch.sum(torch.gt(g_opt_list, self.eps), axis=1) > 0))
      else:
        min_index = torch.argmin(f_opt_list.squeeze())
      x_opt_final = x_opt_list[min_index,:]
      
    # return final solution
    return x_opt_final

  def run_main_loop(self): # TODO: Add in more parameters / options
    # fix random seed
    if self.seed is not None:
      numpy.random.seed(seed=self.seed)
      torch.manual_seed(self.seed)
      botorch.utils.sampling.manual_seed(seed=self.seed)

    # start with some initial random samples
    if self.x_data is None:
        x_next = self.xL + (self.xU - self.xL) * torch.rand(self.Ninit, self.nx, dtype=torch.float64)
    else:
        x_next = self.x_data
    z_next, h_next, fun_next = self.eval_new_samples(x_next)
    self.add_new_samples(x_next, z_next, h_next, fun_next)

    # loop over remaining samples
    for k in range(self.NFE):
      print(f'starting iteration {k+1}')  
      # train models at start of run (given most recent data)
      self.train_models()
      
      # run optimization procedure to get next sample
      x_next = self.optimize_acq()
      x_next = x_next.unsqueeze(0)

      # evaluate functions at new sample
      z_next, h_next, fun_next = self.eval_new_samples(x_next)

      # add new data to history
      self.add_new_samples(x_next, z_next, h_next, fun_next)

  def get_current_best_point(self, iter=None):
    if iter is None:
      iter = self.x_data.shape[0]
    if iter == 0:
      min_index = 0
    else:
      min_index = torch.argmin(self.fun_data[0:iter+1,0] + self.M*(torch.sum(torch.gt(self.fun_data[0:iter+1,1:], self.eps), axis=1) > 0))
    return self.x_data[min_index,:], self.fun_data[min_index,0], self.fun_data[min_index,1:]

  def calculate_simple_regret(self):
    Ntotal = self.NFE + self.Ninit
    regret_penalty = torch.zeros((Ntotal))
    for i in range(Ntotal):
      _, best_obj_i, best_cons_i = self.get_current_best_point(i)
      if self.problem.f_opt is not None:
        regret_plus_i = torch.max( best_obj_i - torch.tensor(self.problem.f_opt), torch.tensor(0) )
      else:
        regret_plus_i = best_obj_i
      violation_plus_i = 0
      for best_cons_i_j in best_cons_i:
        violation_plus_i += torch.max(best_cons_i_j - self.eps, torch.tensor(0))
      regret_penalty[i] = regret_plus_i + violation_plus_i  # add a buffer in to make sure log plot works
    return regret_penalty

  def plot_simple_regret(self):
    regret_penalty = self.calculate_simple_regret()
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.step(numpy.arange(start=1, stop=self.Ninit + self.NFE+1), regret_penalty.numpy())
    ax.set_yscale('log')

  def plot_functions(self, index_list=[0]):
    if self.nx <= 2:
      fun_list = self.fun_list
      ftype = self.fun_type
      for i in index_list:
        if ftype[i] == "wb":
          f = fun_list[i].eval_torch
        elif ftype[i] == "bb":
          f = fun_list[i].eval_cb_torch
        elif ftype[i] == "gb":
          f = fun_list[i].eval_qb_torch
        plot_torch_function(f, self.xL, self.xU)