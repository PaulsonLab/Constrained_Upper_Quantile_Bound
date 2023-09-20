#!/usr/bin/env python
# coding: utf-8

# In[62]:


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


get_ipython().run_line_magic('matplotlib', 'inline')

torch.manual_seed(0)
device = (
    torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:4")
)
dtype = torch.float

print("Using ", device)

class Problem:
    def __init__(self, g, dim, h, A, xL, xU, alpha, f_opt=None, x_opt=None, g_s = None):
      self.g = g
      self.dim = dim
      self.h = h
      self.A = A
      self.xL = xL
      self.xU = xU
      self.alpha = alpha
      self.f_opt = f_opt
      self.x_opt = x_opt
      self.g_s = g_s
x_res = None
g_res = None
c_res = None


# In[63]:


# 2D
# Bazarra et al

def g(x,y):
    x1 = x[...,0]
    x2 = x[...,1]
    y1 = y[...,0]
    y2 = y[...,1]
    g0 =  -1*(2*x1**2 + 2*x2**2 - y2) 
    g1 = -1*(5*x1 + x2 - 5)
    g2 = -1*(y1 - x1)
    return torch.stack((g0, g1, g2),-1)

def h(z):
    z1 = z[...,0]
    z2 = z[...,1]
    y1 = 2*z2**2
    y2 = 2*z1*z2 + 6*z1 + 4*z2
    y1 = y1.unsqueeze(-1)
    y2 = y2.unsqueeze(-1)
    return torch.hstack((y1, y2))

def g_s(x):
    global x_res
    global g_res
    global c_res
    x1 = x[0]
    x2 = x[1]
    g0 = (2*x1**2 + 2*x2**2 - (2*x1*x2 + 6*x1 + 4*x2)) 
    g1 = (5*x1 + x2 - 5)
    g2 = (2*x2**2 - x1)
    cons = numpy.array([g1,g2])
    g_eval = numpy.array([[g0,g1,g2]])
    if x_res is None:
        x_res = numpy.array([x])
        g_res = numpy.array([[g0,g1,g2]])
    else:
        x_res = numpy.append(x_res, [x], axis=0)
        g_res = numpy.append(g_res, g_eval, axis=0)
    return g0 + 1e5*((cons>1e-5)*cons).sum(0)

# grey box fun type list
alpha = 0.95

# mapping from full variable space x to input to composite black-box z --> z = A*x
A = torch.eye(2)
print(A)
# bounds on variables
xL = torch.tensor([0.01, 0.01])
xU = torch.tensor([1,1])
bounds = torch.vstack((xL, xU))
x_opt = [0.86822553, 0.65887234]
f_opt = -6.613085467

# test functions
print(g(torch.Tensor(x_opt), h(torch.Tensor(x_opt))))
print(g_s(x_opt))
print(f_opt)

# create a problem instance
Baz = Problem(g=g, dim=2, h=h, A=A, xL=xL, xU=xU, alpha=alpha, f_opt=f_opt, x_opt=x_opt, g_s = g_s)
x_res = None
g_res = None
c_res = None


# In[64]:


# 3D

#example: spring    

def g(x,y):
  x1 = x[...,0]
  x2 = x[...,1]
  x3 = x[...,2]
  y1 = y[...,0]
  y2 = y[...,1]
  g0 = -1*y1*(2+x3)
  g1 = -1*(2*x2**2 - x1)
  g2 = -1*((4*x2**2 - x1*x2)/(12566*((x1**2*x2)*x1 - x1**4)) + 1/(5108*x1**2) - 1)
  g3 = -1*(1 - 140.45*x1/(y2))
  g4 = -1*((x1 + x2)/1.5 - 1)
  return torch.stack((g0, g1, g2, g3, g4),-1)

def h(z):
  z1 = z[...,0]
  z2 = z[...,1]
  z3 = z[...,2]
  y1 = z1**2*z2
  y2 = z2**3*z3
  y1 = y1.unsqueeze(-1)
  y2 = y2.unsqueeze(-1)
  return torch.hstack((y1, y2))

def g_s(x):
    global x_res
    global g_res
    global c_res
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    g0 = (x1**2*x2)*(2+x3)
    g1 = ((4*x2**2 - x1*x2)/(12566*((x1**2*x2)*x1 - x1**4)) + 1/(5108*x1**2) - 1)
    g2 = (2*x2**2 - x1)
    g3 = (1 - 140.45*x1/(x2**2*x3))
    g4 = ((x1 + x2)/1.5 - 1)
    cons = numpy.array([g1,g2,g3,g4])
    g_eval = numpy.array([[g0,g1,g2,g3,g4]])
    if x_res is None:
        x_res = numpy.array([x])
        g_res = numpy.array([[g0,g1,g2,g3,g4]])
    else:
        x_res = numpy.append(x_res, [x], axis=0)
        g_res = numpy.append(g_res, g_eval, axis=0)
    return g0 + 1e5*((cons>1e-5)*cons).sum(0)

#specify probability level for each function
alpha = 0.95

# mapping from full variable space x to input to composite black-box z --> z = A*x
A = torch.eye(3)
print(A)
# bounds on variables
xL = torch.tensor([0.05, 0.25, 2])
xU = torch.tensor([2, 1.3, 15])
bounds = torch.vstack((xL, xU))

# optimal solution (as list)
f_opt = 0.0126652327883
x_opt = [0.051689156131, 0.356720026419, 11.288831695483]

# test functions
print(g(torch.Tensor(x_opt), h(torch.Tensor(x_opt))))
print(g_s(x_opt))
print(f_opt)

# create a problem instance
Spr = Problem(g=g, dim=3, h=h, A=A, xL=xL, xU=xU, alpha=alpha, f_opt=f_opt, x_opt=x_opt, g_s=g_s)
x_res = None
g_res = None
c_res = None


# In[65]:


# 4D
### Rosen Suzuki

# define problem (g0 = obj, g1,g2,... = constraints)
def g(x, y):
  x1 = x[...,0]
  x2 = x[...,1]
  x3 = x[...,2]
  x4 = x[...,3]
  y1 = y[...,0]
  y2 = y[...,1]
  g0 = -1*(x1**2 + x2**2 + x4**2 - 5*x1 - 5*x2 + y1)
  g1 = (8 - x1**2 - x2**2 - x3**2 - x4**2 - x1 + x2 - x3 + x4)
  g2 = (10 - x1**2 - 2*x2**2 - y2 + x1 + x4)
  g3 = (5 - 2*x1**2 - x2**2 - x3**2 - 2*x1 + x2 + x4)
  return torch.stack((g0,g1,g2,g3),-1)

# unknown black box function
def h(z):
  z1 = z[...,0]
  z2 = z[...,1]
  y1 = 2*z1**2 - 21*z1 + 7*z2
  y2 = z1**2 + 2*z2**2
  return torch.stack((y1, y2),-1)

def g_s(x):
    global x_res
    global g_res
    global c_res
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    g0 = (x1**2 + x2**2 + x4**2 - 5*x1 - 5*x2 + 2*x3**2 - 21*x3 + 7*x4)
    g1 = -(8 - x1**2 - x2**2 - x3**2 - x4**2 - x1 + x2 - x3 + x4)
    g2 = -(10 - x1**2 - 2*x2**2 - (x3**2 + 2*x4**2) + x1 + x4)
    g3 = -(5 - 2*x1**2 - x2**2 - x3**2 - 2*x1 + x2 + x4)
    cons = numpy.array([g1,g2,g3])
    g_eval = numpy.array([[g0,g1,g2,g3]])
    if x_res is None:
        x_res = numpy.array([x])
        g_res = numpy.array([[g0,g1,g2,g3]])
        
    else:
        x_res = numpy.append(x_res, [x], axis=0)
        g_res = numpy.append(g_res, g_eval, axis=0)
        
    return g0 + 1e5*((cons>1e-5)*cons).sum(0)

# specify probability level (holds for each function)
alpha = 0.95

# mapping from full variable space x to input to composite black-box z --> z = A*x
A = torch.zeros((2,4))
A[0,2] = 1.0
A[1,3] = 1.0
print(A)
# bounds on variables
xL = torch.tensor([-2.0, -2.0, -2.0, -2.0])
xU = torch.tensor([2.0, 2.0, 2.0, 2.0])
bounds = torch.vstack((xL, xU))

# optimal solution
f_opt = -44.0
x_opt = [0.0, 1.0, 2.0, -1.0]

# test functions
print(g(torch.Tensor(x_opt), h(torch.Tensor([2, -1]))))
print(g_s(x_opt))
print(f_opt)

# create a problem instance
RS = Problem(g=g, dim=4, h=h, A=A, xL=xL, xU=xU, alpha=alpha, f_opt=f_opt, x_opt=x_opt, g_s = g_s)
x_res = None
g_res = None
c_res = None


# In[67]:


# 5D
# ex211

def g(x,y):
    x1 = x[...,0]
    x2 = x[...,1]
    x3 = x[...,2]
    x4 = x[...,3]
    x5 = x[...,4]
    y1 = y[...,0]
    y2 = y[...,1]
    g0 = -1*(42*x1 - 50*(y1) +44*x2 + 45*x3 + 47*x4 + 47.5*x5)
    g1 = -1*(20*x1 + y2 + 4*x5 - 39)
    return torch.stack((g0,g1),-1)

def h(z):
    z1 = z[...,0]
    z2 = z[...,1]
    z3 = z[...,2]
    z4 = z[...,3]
    z5 = z[...,4]
    y1 = z1**2 + z2**2 + z3**2 + z4**2 + z5**2
    y2 = 12*z2 + 11*z3 + 7*z4
    y1 = y1.unsqueeze(-1)
    y2 = y2.unsqueeze(-1)
    return torch.hstack((y1, y2))

def g_s(x):
    global x_res
    global g_res
    global c_res
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    g0 = (42*x1 - 50*(x1**2 + x2**2 + x3**2 + x4**2 + x5**2) +44*x2 + 45*x3 + 47*x4 + 47.5*x5)
    g1 = (20*x1 + 12*x2 + 11*x3 + 7*x4 + 4*x5 - 39)
    cons = numpy.array([g1])
    g_eval = numpy.array([[g0,g1]])
    if x_res is None:
        x_res = numpy.array([x])
        g_res = numpy.array([[g0,g1]])

    else:
        x_res = numpy.append(x_res, [x], axis=0)
        g_res = numpy.append(g_res, g_eval, axis=0)

    return g0 + 1e5*((cons>1e-5)*cons).sum(0)

#specify probability level for each function
alpha = 0.95

# mapping from full variable space x to input to composite black-box z --> z = A*x
A = torch.eye(5)
print(A)
# bounds on variables
xL = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
xU = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

# optimal solution (as list)
f_opt = -17
x_opt = [1,1,0,1,0]
# test functions
print(g(torch.Tensor(x_opt), h(torch.Tensor(x_opt))))
#print(g_s(x_opt))
#print(f_opt)
print (xL)

# create a problem instance
Ex211 = Problem(g=g, dim=5, h=h, A=A, xL=xL, xU=xU, alpha=alpha, f_opt=f_opt, x_opt=x_opt, g_s=g_s)
x_res = None
g_res = None
c_res = None


# In[68]:


# 6D

# ex212

def g(x,y):
    x1 = x[...,0]
    x2 = x[...,1]
    x3 = x[...,2]
    x4 = x[...,3]
    x5 = x[...,4]
    x6 = x[...,5]
    y1 = y[...,0]
    y2 = y[...,1]
    g0 = (-(-0.5*(x1*x1 + x2*x2 + x3*x3 + x4*x4 + x5*x5) - y1) + 10*x6)
    g1 = -1*(6*x1 + 3*x2 + 3*x3 + 2*x4 + x5 - 6.5)
    g2 = -1*(y2 - 20)
    return torch.stack((g0,g1,g2),-1)

def h(z):
    x1 = z[...,0]
    x2 = z[...,1]
    x3 = z[...,2]
    x4 = z[...,3]
    x5 = z[...,4]
    x6 = z[...,5]
    y1 = 10.5*x1 + 7.5*x2 + 3.5*x3 + 2.5*x4 + 1.5*x5
    y2 = 10*x1 + 10*x3 + x6
    y1 = y1.unsqueeze(-1)
    y2 = y2.unsqueeze(-1)
    return torch.hstack((y1, y2))

def g_s(x):
    global x_res
    global g_res
    global c_res
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    g0 = -1*(-(-0.5*(x1*x1 + x2*x2 + x3*x3 + x4*x4 + x5*x5) - (10.5*x1 + 7.5*x2 + 3.5*x3 + 2.5*x4 + 1.5*x5)) + 10*x6)
    g1 = (6*x1 + 3*x2 + 3*x3 + 2*x4 + x5 - 6.5)
    g2 = (10*x1 + 10*x3 + x6 - 20)
    cons = numpy.array([g1,g2])
    g_eval = numpy.array([[g0,g1,g2]])
    if x_res is None:
        x_res = numpy.array([x])
        g_res = numpy.array([[g0,g1,g2]])
    else:
        x_res = numpy.append(x_res, [x], axis=0)
        g_res = numpy.append(g_res, g_eval, axis=0)
    return g0 + 1e5*((cons>1e-5)*cons).sum(0)

#specify probability level for each function
alpha = 0.95

# mapping from full variable space x to input to composite black-box z --> z = A*x
A = torch.eye(6)

# bounds on variables
xL = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
xU = torch.tensor([30.0, 30.0, 30.0, 30.0, 30.0, 30.0])

# optimal solution (as list)
f_opt = -213
x_opt = [0,1,0,1,1,20]
# test functions
print(g(torch.Tensor(x_opt), h(torch.Tensor(x_opt))))
print(g_s(x_opt))
print(f_opt)
print(A)


# create a problem instance
Ex212 = Problem(g=g, dim=6, h=h, A=A, xL=xL, xU=xU, alpha=alpha, f_opt=f_opt, x_opt=x_opt, g_s=g_s)
x_res = None
g_res = None
c_res = None


# In[70]:


# 7D

def g(x,y):
    x1 = x[...,0]
    x2 = x[...,1]
    x3 = x[...,2]
    x4 = x[...,3]
    x5 = x[...,4]
    x6 = x[...,5]
    x7 = x[...,6]
    y1 = y[...,0]
    y2 = y[...,1]
    g0 = -(y1+x3**4+3*(x4-11)**2+10*x5**6+7*x6**2+x7**4-4*x6*x7-10*x6-8*x7)
    g1 = -(-127 + 2*x1**2 + y2 + 5*x5)
    g2 = -(-282 + 7*x1 + 3*x2 + 10*x3**2 + x4 - x5)
    g3 = -(-196 + 23*x1 + x2**2 + 6*x6**2 - 8*x7)
    g4 = -(4*x1**2 + x2**2 - 3*x1*x2 + 2*x3**2 + 5*x6 - 11*x7)
    return torch.stack((g0,g1,g2,g3,g4),-1)

def h(z):
    x1 = z[...,0]
    x2 = z[...,1]
    x3 = z[...,2]
    x4 = z[...,3]
    y1 = (x1-10)**2+5*(x2-12)**2
    y2 = 3*x2**4 + x3 + 4*x4**2
    y1 = y1.unsqueeze(-1)
    y2 = y2.unsqueeze(-1)
    return torch.hstack((y1, y2))

def g_s(x):
    global x_res
    global g_res
    global c_res
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    x7 = x[6]
    g0 = (x1-10)**2+5*(x2-12)**2+x3**4+3*(x4-11)**2+10*x5**6+7*x6**2+x7**4-4*x6*x7-10*x6-8*x7
    g1 = -127 + 2*x1**2 + 3*x2**4 + x3 + 4*x4**2 + 5*x5
    g2 = -282 + 7*x1 + 3*x2 + 10*x3**2 + x4 - x5
    g3 = -196 + 23*x1 + x2**2 + 6*x6**2 - 8*x7
    g4 = 4*x1**2 + x2**2 - 3*x1*x2 + 2*x3**2 + 5*x6 - 11*x7
    cons = numpy.array([g1,g2,g3,g4])
    g_eval = numpy.array([[g0,g1,g2,g3,g4]])
    if x_res is None:
        x_res = numpy.array([x])
        g_res = numpy.array([[g0,g1,g2,g3,g4]])
    else:
        x_res = numpy.append(x_res, [x], axis=0)
        g_res = numpy.append(g_res, g_eval, axis=0)
    return g0 + 1e5*((cons>1e-5)*cons).sum(0)
    
    
#specify probability level for each function
alpha = 0.95

# mapping from full variable space x to input to composite black-box z --> z = A*x
A = torch.zeros((4,7))
A[0,0] = 1.0
A[1,1] = 1.0
A[2,2] = 1.0
A[3,3] = 1.0
print(A)
# bounds on variables
xL = -10*torch.ones(7)
xU = 10*torch.ones(7)

# optimal solution (as list)
f_opt = 680.63
x_opt = [2.330499, 1.951372, -0.4775414, 4.365726, -0.624487, 1.038131, 1.5942267]
# test functions
print(g(torch.Tensor(x_opt), h(torch.Tensor(x_opt))))
#print(g_s(x_opt))
#print(f_opt)
print(xL)
# create a problem instance
g09 = Problem(g=g, dim=7, h=h, A=A, xL=xL, xU=xU, alpha=alpha, f_opt=f_opt, x_opt=x_opt, g_s = g_s)
x_res = None
g_res = None
c_res = None


# In[71]:


# 8D

def g(x,y):
    x1 = x[...,0]
    x2 = x[...,1]
    x3 = x[...,2]
    x4 = x[...,3]
    x5 = x[...,4]
    x6 = x[...,5]
    x7 = x[...,6]
    x8 = x[...,7]
    y1 = y[...,0]
    y2 = y[...,1]
    y3 = y[...,2]
    g0 = -1*(y3+ 0.4*x2**0.67/x8**0.67 - x1 + 10)
    g1 = -1*(0.0588*x5*x7 + 0.1*x1 - 1)
    g2 = -1*(0.0588*x6*x8 + 0.1*x1 + 0.1*x2 - 1)
    g3 = -1*(4*x3/x5 + 2/(y1) + 0.0588*x7/x3**1.3 - 1)
    g4 = -1*(y2 + 0.0588*x4**1.3*x8 - 1)
    return torch.stack((g0,g1,g2,g3,g4),-1)

def h(x):
    x1 = x[...,0]
    x2 = x[...,1]
    x3 = x[...,2]
    x4 = x[...,3]
    x5 = x[...,4]
    x6 = x[...,5]
    x7 = x[...,6]
    y1 = x3**0.71*x5
    y2 = 4*x4/x6 + 2/(x4**0.71*x6)
    y3 = 0.4*x1**0.67/x7**0.67 - x2
    y1 = y1.unsqueeze(-1)
    y2 = y2.unsqueeze(-1)
    y3 = y3.unsqueeze(-1)
    return torch.hstack((y1, y2, y3))

def g_s(x):
    global x_res
    global g_res
    global c_res

    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    x7 = x[6]
    x8 = x[7]
    g0 = ((0.4*x1**0.67/x7**0.67 + 0.4*x2**0.67/x8**0.67 - x1 - x2) + 10)
    g1 = (0.0588*x5*x7 + 0.1*x1 - 1)
    g2 = (0.0588*x6*x8 + 0.1*x1 + 0.1*x2 - 1)
    g3 = (4*x3/x5 + 2/(x3**0.71*x5) + 0.0588*x7/x3**1.3 - 1)
    g4 = (4*x4/x6 + 2/(x4**0.71*x6) + 0.0588*x4**1.3*x8 - 1)
    cons = numpy.array([g1,g2,g3,g4])
    g_eval = numpy.array([[g0,g1,g2,g3,g4]])
    if x_res is None:
        x_res = numpy.array([x])
        g_res = g_eval
    else:
        x_res = numpy.append(x_res, [x], axis=0)
        g_res = numpy.append(g_res, g_eval, axis=0)

    return g0 + 1e5*((cons>1e-5)*cons).sum(0)

#specify probability level for each function
alpha = 0.95

# mapping from full variable space x to input to composite black-box z --> z = A*x
A = torch.zeros((7,8))
A[0,0] = 1.0
A[1,1] = 1.0
A[2,2] = 1.0
A[3,3] = 1.0
A[4,4] = 1.0
A[5,5] = 1.0
A[6,6] = 1.0
# bounds on variables
xL = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
xU = torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

# optimal solution (as list)
f_opt = 3.91801023
x_opt = [6.345784434113570, 2.341009418806070, 0.670868568115071, 0.534745716696259, 5.952793514985580, 5.316398716636410, 1.043989230843780, 0.420085834738904]

# test functions
print(g(torch.Tensor(x_opt), h(torch.Tensor(x_opt))))
print(g_s(x_opt))
print(f_opt)
print(A)
# create a problem instance
Ex724 = Problem(g=g, dim=8, h=h, A=A, xL=xL, xU=xU, alpha=alpha, f_opt=f_opt, x_opt=x_opt, g_s = g_s)
x_res = None
g_res = None
c_res = None


# In[72]:


# 10D

def g(x,y):
    x1 = x[...,0]
    x2 = x[...,1]
    x3 = x[...,2]
    x4 = x[...,3]
    x5 = x[...,4]
    x6 = x[...,5]
    x7 = x[...,6]
    x8 = x[...,7]
    x9 = x[...,8]
    x10 = x[...,9]
    y1 = y[...,0]
    y2 = y[...,1]
    y3 = y[...,2]
    y4 = y[...,3]
    g0 = (48*x1 - 0.5*(y1 + 100*x5*x5 + 100*x6*x6 + 100*x7*x7 + 100*x8*x8 + 100*x9*x9 + 100*x10*x10) + 42*x2 +\
           y3 + 47*x7 + 42*x8 + 45*x9 + 46*x10)
    g1 = y2 - 2*x7 - 6*x8 - 2*x9 - 2*x10 + 4
    g2 = 6*x1 - 5*x2 + 8*x3 - 3*x4 + x6 + 3*x7 + 8*x8 + 9*x9 - 3*x10 - 22
    g3 = -5*x1 + 6*x2 + 5*x3 + 3*x4 + 8*x5 - 8*x6 + 9*x7 + 2*x8 - 9*x10 + 6
    g4 = y4 + 3*x7 - 9*x8 - 9*x9 - 3*x10 + 23
    g5 = -8*x1 + 7*x2 - 4*x3 - 5*x4 - 9*x5 + x6 - 7*x7 - x8 + 3*x9 - 2*x10 + 12
    return torch.stack((-1*g0,-1*g1,-1*g2,-1*g3,-1*g4,-1*g5),-1)

def h(z):
    x1 = z[...,0]
    x2 = z[...,1]
    x3 = z[...,2]
    x4 = z[...,3]
    x5 = z[...,4]
    x6 = z[...,5]
    y1 = 100*x1*x1 + 100*x2*x2 + 100*x3*x3 + 100*x4*x4
    y2 = -2*x1 - 6*x2 - x3 - 3*x5 - 3*x6
    y3 =  48*x3 + 45*x4 + 44*x5 + 41*x6 
    y4 = 9*x1 + 5*x2 - 9*x4 + x5 - 8*x6
    y1 = y1.unsqueeze(-1)
    y2 = y2.unsqueeze(-1)
    y3 = y3.unsqueeze(-1)
    y4 = y4.unsqueeze(-1)
    return torch.hstack((y1, y2, y3, y4))

def g_s(x):
    global x_res
    global g_res
    global c_res
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    x7 = x[6]
    x8 = x[7]
    x9 = x[8]
    x10 = x[9]
    g0 = (48*x1 - 0.5*(100*x1*x1 + 100*x2*x2 + 100*x3*x3 + 100*x4*x4 + 100*x5*x5 + 100*x6*x6 + 100*x7*x7 + 100*x8*x8 + 100*x9*x9 + 100*x10*x10) + 42*x2 +\
           48*x3 + 45*x4 + 44*x5 + 41*x6 + 47*x7 + 42*x8 + 45*x9 + 46*x10)
    g1 = -2*x1 - 6*x2 - x3 - 3*x5 - 3*x6 - 2*x7 - 6*x8 - 2*x9 - 2*x10 + 4
    g2 = 6*x1 - 5*x2 + 8*x3 - 3*x4 + x6 + 3*x7 + 8*x8 + 9*x9 - 3*x10 - 22
    g3 = -5*x1 + 6*x2 + 5*x3 + 3*x4 + 8*x5 - 8*x6 + 9*x7 + 2*x8 - 9*x10 + 6
    g4 = 9*x1 + 5*x2 - 9*x4 + x5 - 8*x6 + 3*x7 - 9*x8 - 9*x9 - 3*x10 + 23
    g5 = -8*x1 + 7*x2 - 4*x3 - 5*x4 - 9*x5 + x6 - 7*x7 - x8 + 3*x9 - 2*x10 + 12
    cons = numpy.array([g1,g2,g3,g4,g5])
    g_eval = numpy.array([[g0,g1,g2,g3,g4,g5]])
    if x_res is None:
        x_res = numpy.array([x])
        g_res = g_eval
    else:
        x_res = numpy.append(x_res, [x], axis=0)
        g_res = numpy.append(g_res, g_eval, axis=0)
    return g0 + 1e5*((cons>1e-5)*cons).sum(0)

#specify probability level for each function
alpha = 0.95

# mapping from full variable space x to input to composite black-box z --> z = A*x
A = torch.zeros(6,10)
A[0,0] = 1.0
A[1,1] = 1.0
A[2,2] = 1.0
A[3,3] = 1.0
A[4,4] = 1.0
A[5,5] = 1.0
print(A)
# bounds on variables
xL = torch.zeros(10)
xU = torch.ones(10)

# optimal solution (as list)
f_opt = -39
x_opt = [1,0,0,1,1,1,0,1,1,1]

# test functions
print(g(torch.Tensor(x_opt), h(torch.Tensor(x_opt))))
#print(g_s(x_opt))
print(f_opt)

# create a problem instance
Ex216 = Problem(g=g, dim=10, h=h, A=A, xL=xL, xU=xU, alpha=alpha, f_opt=f_opt, x_opt=x_opt, g_s = g_s)
x_res = None
g_res = None
c_res = None


# In[73]:


# 4D st_bpv1


def g(x,y):
    x1 = x[...,0]
    x2 = x[...,1]
    x3 = x[...,2]
    x4 = x[...,3]
    y1 = y[...,0]
    y2 = y[...,1]
    y3 = y[...,2]
    g0 = -1*(y1 + x2*x4)
    g1 = -1*(30 - y2)
    g2 = -1*(20 - y3)
    g3 = -1*(10 + 1.6667*x3 - x4)
    g4 = -1*(x3 + x4 - 15)
    return torch.stack((g0,g1,g2,g3,g4),-1)

def h(z):
    x1 = z[...,0]
    x2 = z[...,1]
    x3 = z[...,2]
    y1 = x1*x3
    y2 = x1 + 3*x2
    y3 = 2*x1 + x2
    y1 = y1.unsqueeze(-1)
    y2 = y2.unsqueeze(-1)
    y3 = y3.unsqueeze(-1)
    return torch.hstack((y1, y2, y3))

def g_s(x):
    global x_res
    global g_res
    global c_res
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    g0 = x1*x3 + x2*x4
    g1 = 30 - (x1 + 3*x2)
    g2 = 20 - (2*x1 + x2)
    g3 = 10 + 1.6667*x3 - x4
    g4 = x3 + x4 - 15
    cons = numpy.array([g1,g2,g3])
    g_eval = numpy.array([[g0,g1,g2,g3]])
    if x_res is None:
        x_res = numpy.array([x])
        g_res = g_eval

    else:
        x_res = numpy.append(x_res, [x], axis=0)
        g_res = numpy.append(g_res, g_eval, axis=0)

    return g0 + 1e5*((cons>1e-5)*cons).sum(0)

#specify probability level for each function
alpha = 0.95

# mapping from full variable space x to input to composite black-box z --> z = A*x
A = torch.zeros(3,4)
A[0,0] = 1.0
A[1,1] = 1.0
A[2,2] = 1.0

# bounds on variables
xL = torch.zeros(4)
xU = torch.tensor([27,16,10,10])

# optimal solution (as list)
f_opt = 10
x_opt = [27,1,0,10]

# test functions
print(g(torch.Tensor(x_opt), h(torch.Tensor(x_opt))))
#print(g_s(x_opt))
print(f_opt)
print(A)
# create a problem instance
bpv4D = Problem(g=g, dim=4, h=h, A=A, xL=xL, xU=xU, alpha=alpha, f_opt=f_opt, x_opt=x_opt, g_s = g_s)
x_res = None
g_res = None
c_res = None


# In[74]:


# 3D Ex314

def g(x,y):
    x1 = x[...,0]
    x2 = x[...,1]
    x3 = x[...,2]
    y1 = y[...,0]
    y2 = y[...,1]
    g0 = y2
    g1 = -(x1*(y1)+x2*(2*x2-2*x1-x3)+x3*(2*x1-x2+2*x3)-20*x1+9*x2-13*x3)-24
    g2 = x1+x2+x3-4
    g3 = 3*x2+x3-6
    return torch.stack((-1*g0,-1*g1,-1*g2,-1*g3),-1)

def h(z):
    x1 = z[...,0]
    x2 = z[...,1]
    x3 = z[...,2]
    y1 = 4*x1-2*x2+2*x3
    y2 = x2 - x3 - 2*x1
    y1 = y1.unsqueeze(-1)
    y2 = y2.unsqueeze(-1)
    return torch.hstack((y1, y2))

def g_s(x):
    global x_res
    global g_res
    global c_res
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    g0 = x2 - x3 - 2*x1
    g1 = -(x1*(4*x1-2*x2+2*x3)+x2*(2*x2-2*x1-x3)+x3*(2*x1-x2+2*x3)-20*x1+9*x2-13*x3)-24
    g2 = x1+x2+x3-4
    g3 = 3*x2+x3-6
    cons = numpy.array([g1,g2,g3])
    g_eval = numpy.array([[g0,g1,g2,g3]])
    if x_res is None:
        x_res = numpy.array([x])
        g_res = g_eval

    else:
        x_res = numpy.append(x_res, [x], axis=0)
        g_res = numpy.append(g_res, g_eval, axis=0)

    return g0 + 1e5*((cons>1e-5)*cons).sum(0)

#specify probability level for each function
alpha = 0.95

# mapping from full variable space x to input to composite black-box z --> z = A*x
A = torch.eye(3)
print(A)
# bounds on variables
xL = torch.tensor([-2.0, 0.0, -3.0,])
xU = torch.tensor([2.0, 6.0, 3.0])

# optimal solution (as list)
f_opt = -4.0
x_opt = [0.5, 0.0, 3.0]

# test functions
print(g(torch.Tensor(x_opt), h(torch.Tensor(x_opt))))
print(g_s(x_opt))
print(f_opt)

# create a problem instance
Ex314 = Problem(g=g, dim=3, h=h, A=A, xL=xL, xU=xU, alpha=alpha, f_opt=f_opt, x_opt=x_opt, g_s = g_s)
global x_res
global g_res
global c_res


# In[20]:


# generate random points
def gen_rand_points(bounds, num_samples):
    points_nlzd = torch.rand(num_samples, bounds.shape[-1]).to(bounds)
    return bounds[0] + (bounds[1] - bounds[0]) * points_nlzd

# training data
# train_X = gen_rand_points(bounds, n_init)
# train_Z = torch.matmul(train_X, A.T)
# train_Y = h(train_Z)

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

# # custom likelihood
# _, aug_batch_shape = HigherOrderGP.get_batch_dimensions(train_X=train_Z, train_Y=train_Y)
# likelihood = GaussianLikelihood(batch_shape=aug_batch_shape, noise_constraint=Interval(lower_bound=1e-5, upper_bound=1e-4))
# mll = ExactMarginalLogLikelihood(likelihood, model_hogp)
# fit_gpytorch_model(mll)

# train model


# In[21]:


class UpperConstrainedQuantile(MCAcquisitionFunction):
    def __init__(
        self,
        h: Model,
        A: torch.tensor,
        g = None,
        alpha = 0.95,
        num_samples = 50,
        regularization_strength = 0.1,
        sampler: Optional[MCSampler] = None,
        penalty_weight = 1e4,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super(MCAcquisitionFunction, self).__init__(model=h)
        self.A = A
        self.num_samples = num_samples
        self.g = g
        self.alpha = alpha
        self.regularization_strength = regularization_strength
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_samples]))
        self.sampler = sampler
        self.penalty_weight = penalty_weight

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate Upper Quantile Function with Constraints on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            Tensor: A `(b)`-dim Tensor of Quantile values at the
                given design points `X`.
        """
        A = self.A
        g = self.g

        # get posterior samples of h(Z)
        X = X
        Z = torch.bmm(X, A.T.repeat((X.shape[0],1,1)))
        posterior = self.model.posterior(Z)
        Y = self.sampler(posterior)                     # n x b x q x y  (output samples)
        Y = Y.squeeze(dim=-2)                           # n x b x y      (remove q since it is always = 1)

        # put samples through function
        X = X.squeeze(dim=1)                            # b x d (remove q)
        X = X.repeat((self.num_samples,1,1))            # n x b x d (repeat for number of samples)
        G = self.g(X, Y)                                # n x b x o (total number of outputs obj + cons)
        
        # calculate quantiles
        obj = G[...,0]
        con = torch.sum(torch.min(G[:,:,1:],torch.zeros(G.shape[0],G.shape[1],G.shape[2]-1))**2, dim=2, dtype=torch.double)
        quants = obj - self.penalty_weight*con
        soft_sort_G = soft_sort(quants.T, regularization_strength=self.regularization_strength) # b x n
        soft_sort_G = soft_sort_G.T # n x b
        quantile = soft_sort_G[int(self.alpha*self.num_samples)-1,:].squeeze() # b
        # need to reduce samples along n dimension for each batch and output
        #G = torch.max(G,dim=0)[0]                       # b x o
        
        # evaluate a penalty on violated constraints
        #quantile = G[:,0] # b
        #if G.shape[-1] > 1:
          #cons_penalty = torch.sum(torch.min(G[:,1:],torch.zeros(G.shape[0],G.shape[-1]-1))**2,dim=1) # b
          #quantile = quantile - self.penalty_weight*cons_penalty # b
        return quantile
    


# In[22]:


# test for 2D
problem = Baz
batch_size = 1
torch.manual_seed(10)
penalty_weight = 1e5
A = problem.A
bounds = torch.vstack((problem.xL, problem.xU))
n_init = int(2*problem.dim + 1)
train_X = gen_rand_points(bounds, n_init)
train_Z = torch.matmul(train_X, A.T)
train_Y = problem.h(train_Z)
G_curr = problem.g(train_X, train_Y)

model_gb = train_model(
          train_Z,
          train_Y)
    
model_bb = train_model(
          train_X,
          G_curr)


x_test = torch.rand(1,1,2)


# In[23]:


uqbxx = UpperConstrainedQuantile(model_gb, problem.A, problem.g)
uqbxx(x_test)


# In[24]:


model_gb(x_test).mean


# In[25]:


class ConstrainedExpectedImprovementComposite(MCAcquisitionFunction):
    def __init__(
        self,
        h: Model,
        A: torch.tensor,
        g,
        best_f,
        num_samples = 50,
        eta = 1e-4,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super(MCAcquisitionFunction, self).__init__(model=h)
        self.A = A
        self.g = g
        self.best_f = best_f
        self.num_samples = num_samples
        self.eta = eta
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_samples]))
        self.sampler = sampler

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate EIC-CF on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            Tensor: A `(b)`-dim Tensor of Quantile values at the
                given design points `X`.
        """
        A = self.A
        g = self.g

        # get posterior samples of h(Z)
        X = X
        Z = torch.bmm(X, A.T.repeat((X.shape[0],1,1)))
        posterior = self.model.posterior(Z)
        Y = self.sampler(posterior)                     # n x b x q x y  (output samples)
        Y = Y.squeeze(dim=-2)                           # n x b x y      (remove q since it is always = 1)

        # put samples through function
        X = X.squeeze(dim=1)                            # b x d (remove q)
        X = X.repeat((self.num_samples,1,1))            # n x b x d (repeat for number of samples)
        G = self.g(X, Y)                                # n x b x o (total number of outputs obj + cons)

        # get the objective values and calculate improvement
        obj = G[...,0]                                                    # n x b
        improv = (obj - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)   # n x b

        # create soft constraint function that approximates indicator
        def soft_eval_constraint(rhs, eta):
          return torch.sigmoid(rhs / eta)

        # loop over constraints and multiply by feasibility penalty
        num_cons = G.shape[-1] - 1
        for i in range(num_cons):
          cons_eval = soft_eval_constraint(G[...,i+1], self.eta)
          improv = improv.mul(cons_eval)

        # take the average over all samples
        eiccf = improv.mean(dim=0) # b

        return eiccf
    
    


# In[26]:


# test eic-cf
eiccfxx = ConstrainedExpectedImprovementComposite(model_gb, problem.A, problem.g, torch.max(G_curr[...,0]))
eiccfxx(x_test)


# In[27]:


tt = x_test[0,...]
pos = model_bb(tt)
pos2 = model_gb(tt)
print(pos.mean)
print(problem.g(tt, problem.h(tt)))
print(torch.max(G_curr[...,0]))


# In[28]:


# EPBO acquisition
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform

class ExactPenalty(AnalyticAcquisitionFunction):
    def __init__(
        self,
        h: Model,
        beta = 4,
        penalty = 1e4,
        maximize: bool = True,
    ) -> None:
        r"""Single-outcome Exact Penalty Acq.

        Args:
            model: A fitted multi-output model
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem. Note
                that if `maximize=False`, the posterior mean is negated. As a
                consequence `optimize_acqf(PosteriorMean(gp, maximize=False))`
                actually returns -1 * minimum of the posterior mean.
        """
        super(AnalyticAcquisitionFunction, self).__init__(model=h)
        self.posterior_transform = None
        self.beta = beta
        self.maximize = maximize
        self.ro = penalty
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the posterior mean on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of penalized values at the
            given design points `X`.
        """
        mean, sigma = self._mean_and_sigma(X)
        obj = mean[...,0] + self.beta**0.5 * sigma[...,0]
        num_cons = mean.shape[-1] - 1
        vio = 0
        for i in range(num_cons):
            vio = vio + ((mean[...,i+1] + self.beta**0.5 *sigma[...,i+1]) < 1e-4)*(mean[...,i+1] + self.beta**0.5 *sigma[...,i+1]) 
                                                                
        #print(mean)
        #print(sigma)
        #print(obj)
        return obj - self.ro*vio


# In[29]:


#test epbo
epboxx = ExactPenalty(model_bb)
epboxx(x_test)


# In[ ]:





# In[30]:


def optimize_ucq(lcq, bounds, **options):
    cands_nlzd, _ = optimize_acqf(lcq, bounds, **options)
    return cands_nlzd


# In[75]:


from skquant.opt import minimize
import pybobyqa
import cma
import scipy

def run_main_loop(problem, ninit=2, alpha=0.95, ro=50, n_batches=25, opt_tol=1e-3, seed=0, met='uqb',cons=False, x_data=None):
    batch_size = 1
    torch.manual_seed(seed)
    penalty_weight = ro
    A = problem.A

    bounds = torch.vstack((problem.xL, problem.xU))
    n_ninit = int(ninit*problem.dim + 1)
    
    if x_data is None:
        train_X = gen_rand_points(bounds, n_init)
    else:
        train_X = x_data
    train_X = train_X
    train_Z = torch.matmul(train_X, A.T)
    train_Y = problem.h(train_Z)

    # run the BO loop
    max_iter = int(n_batches - ninit*problem.dim - 1)
    for i in range(max_iter):

        tic = time.monotonic()

        # get best observations, log status
        G_curr = problem.g(train_X, train_Y)
        best_f = G_curr[:,0]
        if G_curr.shape[-1] > 1:
            best_f -= penalty_weight * torch.sum(torch.min(G_curr[:,1:],torch.zeros(G_curr.shape[0],G_curr.shape[-1]-1))**2,dim=1)
            best_f = best_f.max().detach()
            #print(f"It {i+1:>2}/{n_batches}, best obs.: {best_f}")

        # Train model
        if met == 'uqb' or met == 'ei-cf':
            greybox = 0
        else:
            greybox = 1

            
        if greybox == 0:
            g0 = G_curr[:,0]
            cons = G_curr[:,1:]
            f_pen = g0 + 1e5*((cons<1e-5)*cons).sum(-1)
            model_h = train_model(
                      train_Z,
                      train_Y)
        else:
            g0 = G_curr[:,0]
            cons = G_curr[:,1:]
            f_pen = g0 + 1e5*((cons<1e-5)*cons).sum(-1)
            model_h = train_model(
                      train_X,
                      G_curr) 
        global x_res
        global g_res
        global c_res
        x_res = None
        g_res = None
        c_res = None
        xL = problem.xL
        xU = problem.xU
        # optimize acquisition
        if met == 'uqb':
            acq = UpperConstrainedQuantile(model_h, problem.A, problem.g, problem.alpha) 
        elif met == 'eic':
            incumbent = torch.max(f_pen)
            num_cons = G_curr.shape[-1] - 1
            for i in num_cons:
                if i == 0:
                    n_c = {i+1: (0.0, None)}
                else:
                    n_c.update({i+1: (0.0, None)})
            acq = botorch.acquisition.analytic.ConstrainedExpectedImprovement(model_h, incumbent, 0, n_c)
        elif met == 'epbo':
            acq = ExactPenalty(model_h)
        elif met == 'ei-cf':
            incumbent = torch.max(f_pen)
            acq = ConstrainedExpectedImprovementComposite(model_h, problem.A, problem.g, incumbent)
        elif met == 'snobfit':
            obj_func = problem.g_s
            xL = problem.xL
            xU = problem.xU
            bounds = numpy.array([xL.numpy(), xU.numpy()], dtype=float).T
            x0 = numpy.array(train_X[0,:])
            budget=int(5*n_batches)
            result, history = minimize(obj_func, x0, bounds, budget, method='snobfit')
            train_X = torch.Tensor(x_res)
            G_curr = torch.Tensor(g_res)
            wall_time = [0]
            break
        elif met == 'cmaes':
            fun = problem.g_s
            x0 = numpy.array(train_X[0,:])
            cma.fmin2(fun, x0, 1,{'bounds':[problem.xL, problem.xU]})
            
            train_X = torch.Tensor(x_res)
            G_curr = torch.Tensor(g_res)
            wall_time = [0]
            break
        elif met == 'bobyqa':
            obj_func = problem.g_s
            x0 = numpy.array(train_X[0,:])
            #print(x0)
            bounds = (problem.xL.numpy(), problem.xU.numpy())
            soln = pybobyqa.solve(obj_func, x0, bounds = bounds,  maxfun=int(5*n_batches),rhobeg=0.05)
            #print(soln)
            train_X = torch.Tensor(x_res)
            G_curr = torch.Tensor(g_res)
            wall_time = [0]
            break
        elif met == 'direct':
            func = problem.g_s
            bounds = scipy.optimize.Bounds(problem.xL.numpy(), problem.xU.numpy())
            maxfun = n_batches
            scipy.optimize.direct(func=func, bounds=bounds, maxfun=maxfun)
            train_X = torch.Tensor(x_res)
            G_curr = torch.Tensor(g_res)
            wall_time = [0]
            break

        # construct acquisition
        #ucq = UpperConstrainedQuantile(model_h, problem.A, problem.g, problem.alpha, penalty_weight)

        # optimize acquisition
        if met == 'uqb' or met=='eic' or met=='epbo' or met=='ei-cf':
          try:
            cands, acq_value = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=3, raw_samples=8192)
            zcands = torch.matmul(cands, A.T)
          except:
            # do random sample
            X_rand = gen_rand_points(bounds, int(2000*problem.dim))
            ucq_val = acq(X_rand[:,None,:])
            loc = torch.argmax(ucq_val)
            cands = X_rand[loc,:].unsqueeze(0)
            zcands = torch.matmul(cands, A.T)
            print('resorting to random sample')

        # make observations and update data
        if cands.shape[0] > 0:
            train_X = torch.cat([train_X, cands])
            train_Z = torch.cat([train_Z, zcands])
            train_Y = torch.cat([train_Y, problem.h(zcands)])

        if i == 0:
            wall_time = numpy.array([time.monotonic() - tic])
        else:
            wall_time = numpy.append(wall_time, time.monotonic() - tic)
        #print(f"Wall time: {time.monotonic() - tic:1f}")

    return train_X, G_curr, wall_time


# In[ ]:


# run for alpha = 0.95, penalty weight in the range of {100, 1000, and 1e4}

problem_list = [Baz, Spr, RS, Ex211, Ex212, g09, Ex724, Ex216, bpv4D, Ex314]
solver_list = [uqb','eic','ei-cf', 'epbo','snobfit', 'cmaes', 'bobyqa', 'direct']
replications = 10
n_batches = 101

def calculate_regret(f_data,maximize=False):
    for i in range(len(f_data)):
        if i == 0:
            regret = numpy.array([f_data[i]])
        else:
            if maximize is False:
                if f_data[i] < regret[-1]:
                    regret = numpy.append(regret, f_data[i])
                else:
                    regret = numpy.append(regret, regret[-1])
            else:
                if f_data[i] > regret[-1]:
                    regret = numpy.append(regret, f_data[i])
                else:
                    regret = numpy.append(regret, regret[-1])
    return regret

for i in range(len(solver_list)):
    i = i
    for j in range(len(problem_list)):
        j = 4
        for k in range(replications):
            if solver_list[i] == 'uqb':
                X_data, G_data, wall_time = run_main_loop(problem=problem_list[0], ninit=2, alpha=0.95, ro=1e5, n_batches=n_batches, opt_tol=1e-5, seed=(j+9)*(k+9),met=solver_list[i])
                pd.DataFrame(X_data.numpy()).to_csv(f'Data/Constrained/X_problem-{j}_solver-{i}_rep-{k}.csv')
                pd.DataFrame(G_data.numpy()).to_csv(f'Data/Constrained/G_problem-{j}_solver-{i}_rep-{k}.csv')
                pd.DataFrame(wall_time).to_csv(f'Data/Constrained/time_problem-{j}_solver-{i}_rep-{k}.csv')
                print(f'finished iteration {k+1} and problem {j+1} for solver {i+1}')  
            else:
                mc = int(2*problem_list[j].dim + 1)
                x_data = pd.read_csv(f'Data/Constrained_v2/X_problem-{j}_solver-0_rep-{k}.csv')
                x0 = torch.Tensor(numpy.squeeze(x_data[:mc].to_numpy()[:,1:]))
                X_data, G_data, wall_time = run_main_loop(problem=problem_list[j], ninit=2, alpha=0.95, ro=1e5, n_batches=n_batches, opt_tol=1e-5, seed=(j+9)*(k+9),met=solver_list[i],x_data=x0)
                pd.DataFrame(G_data).to_csv(f'Data/Constrained_v2/G_problem-{j}_solver-{i}_rep-{k}.csv')
                pd.DataFrame(wall_time).to_csv(f'Data/Constrained_v2/time_problem-{j}_solver-{i}_rep-{k}.csv')
                pd.DataFrame(X_data.numpy()).to_csv(f'Data/Constrained_v2/X_problem-{j}_solver-{i}_rep-{k}.csv')
                print(f'finished iteration {k+1} and problem {j+1} for solver {i+1}')  


# In[ ]:




