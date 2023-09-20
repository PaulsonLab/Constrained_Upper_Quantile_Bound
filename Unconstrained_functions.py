import math
import os
import time
from functools import partial

import botorch
import gpytorch
import numpy
import torch
import scipy
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.constraints.constraints import Interval
from botorch import fit_gpytorch_model
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.models import SingleTaskGP
from cyipopt import minimize_ipopt
from fast_soft_sort.pytorch_ops import soft_sort
from scipy.optimize import minimize
from scipy.stats import norm
import pandas as pd
from typing import Optional
from botorch.optim import optimize_acqf
from fast_soft_sort.pytorch_ops import soft_sort
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform
from torch import Tensor

torch.manual_seed(0)
device = (
    torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:4")
)
dtype = torch.float

print("Using ", device)

class Problem:
    def __init__(self, g, dim, h, A, xL, xU, alpha, f_opt=None, x_opt=None,g_sf = None):
      self.g = g
      self.dim = dim
      self.h = h
      self.A = A
      self.xL = xL
      self.xU = xU
      self.alpha = alpha
      self.f_opt = f_opt
      self.x_opt = x_opt
      self.g_sf = g_sf
x_res = None
g_res = None

# Booth 2d
def g(x,y):
    x1 = x[...,0]
    x2 = x[...,1]
    y1 = y[...,0]
    g0 = y1 + (2*x1 + x2 - 5)**2
    return -1*g0

def h(x):
    x1 = x[...,0]
    x2 = x[...,1]
    y = (x1 +2*x2 - 7)**2
    return y.unsqueeze(-1)

def g_sf(x):
    global x_res
    global g_res
    g0 = (x[0] +2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
    if x_res is None:
        x_res = numpy.array([x])
        g_res = numpy.array([g0])
    else:
        x_res = numpy.append(x_res, [x], axis=0)
        g_res = numpy.append(g_res, [g0], axis=0)
    return g0

# number of initial points
n_init = 2

# specify probability level (holds for each function)
alpha_list = 0.95

# mapping from full variable space x to input to composite black-box z --> z = A*x
A = torch.eye(2,2)

# bounds on variables
xL = torch.tensor([-10.0, -10.0])
xU = torch.tensor([10.0, 10.0])
bounds = torch.vstack((xL, xU))

# optimal solution
f_opt = 0
x_opt = torch.Tensor([1,3])

print(g(x_opt, h(x_opt)))
print(f_opt)
print(x_opt)

# create a problem instance
Booth2D = Problem(g, n_init, h, A, xL, xU, alpha_list, f_opt, x_opt,g_sf)

