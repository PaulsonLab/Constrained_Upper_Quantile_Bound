#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


get_ipython().run_line_magic('matplotlib', 'inline')

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


# In[2]:


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


fun_list = [g]
fun_type = ['gb']

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


# In[3]:


# Extended Rastrigin
def g(x,y):
    x1 = x[...,0]
    x2 = x[...,1]
    x3 = x[...,2]
    y1 = y[...,0]
    y2 = y[...,1]
    g0 = y1 + y2 + 30 + x3**2 - 10*torch.cos(2*3.14159265*x3)
    return -1*g0

def h(x):
    x1 = x[...,0]
    x2 = x[...,1]
    y1 = x1**2 - 10*torch.cos(2*3.14159265*x1)
    y2 = x2**2 - 10*torch.cos(2*3.14159265*x2)
    return torch.stack((y1, y2),-1)

def g_sf(x):
    global x_res
    global g_res
    g0 = x[0]**2 - 10*numpy.cos(2*3.14159265*x[0]) + x[1]**2 - 10*numpy.cos(2*3.14159265*x[1]) + 30 + x[2]**2 - 10*numpy.cos(2*3.14159265*x[2])
    if x_res is None:
        x_res = numpy.array([x])
        g_res = numpy.array([g0])
    else:
        x_res = numpy.append(x_res, [x], axis=0)
        g_res = numpy.append(g_res, [g0], axis=0)
    return g0

fun_list = [g]
fun_type = ['gb']

# number of initial points
n_init = 3

# specify probability level (holds for each function)
alpha_list = 0.95

# mapping from full variable space x to input to composite black-box z --> z = A*x
A = torch.zeros(2,3)
A[0,0] = 1.0
A[1,1] = 1.0

# bounds on variables
xL = torch.tensor([-5.0, -5.0, -5.0])
xU = torch.tensor([5.0, 5.0, 5.0])
bounds = torch.vstack((xL, xU))

# optimal solution
f_opt = 0
x_opt = torch.Tensor([0,0,0])

print(g(x_opt, h(x_opt)))
print(f_opt)
print(x_opt)

# create a problem instance
Ras3D = Problem(g, n_init, h, A, xL, xU, alpha_list, f_opt, x_opt, g_sf)


# In[4]:


# Colville function
def g(x,y):
    x1 = x[...,0]
    x2 = x[...,1]
    x3 = x[...,2]
    x4 = x[...,3]
    y1 = y[...,0]
    g0 = y1 + 90*(x3**2-x4)**2+10.1*((x2-1)**2+(x4-1)**2)+19.8*(x2-1)*(x4-1)
    return -1*g0

def h(x):
    x1 = x[...,0]
    x2 = x[...,1]
    x3 = x[...,2]
    y = 100*(x1**2 - x2)**2 + (x3 - 1)**2 + (x1 - 1)**2
    return y.unsqueeze(-1)


def g_sf(x):
    global x_res
    global g_res
    g0 = 100*(x[0]**2 - x[1])**2 + (x[2] - 1)**2 + (x[0] - 1)**2 + 90*(x[2]**2-x[3])**2+10.1*((x[1]-1)**2+(x[3]-1)**2)+19.8*(x[1]-1)*(x[3]-1)
    if x_res is None:
        x_res = numpy.array([x])
        g_res = numpy.array([g0])
    else:
        x_res = numpy.append(x_res, [x], axis=0)
        g_res = numpy.append(g_res, [g0], axis=0)
    return g0

fun_list = [g]
fun_type = ['gb']

# number of initial points
n_init = 4

# specify probability level (holds for each function)
alpha_list = 0.95

# mapping from full variable space x to input to composite black-box z --> z = A*x
A = torch.zeros(3,4)
A[0,0] = 1.0
A[1,1] = 1.0
A[2,2] = 1.0

# bounds on variables
xL = torch.tensor([-10.0, -10.0,-10.0,-10.0])
xU = torch.tensor([10.0, 10.0,10.0, 10.0])
bounds = torch.vstack((xL, xU))

# optimal solution
f_opt = 0
x_opt = torch.Tensor([1,1,1,1])

print(g(x_opt, h(x_opt)))
print(f_opt)

# create a problem instance
Col4D = Problem(g, n_init, h, A, xL, xU, alpha_list, f_opt, x_opt, g_sf)


# In[5]:


# Dolan's Function

def g(x,y):
  x1 = x[...,0]
  x2 = x[...,1]
  x3 = x[...,2]
  x4 = x[...,3]
  x5 = x[...,4]
  y1 = y[...,0]
  y2 = y[...,1]
  g0 = y1-y2+0.2*x5**2-x2-1
  return -1*g0

def h(x):
  x1 = x[...,0]
  x2 = x[...,1]
  x3 = x[...,2]  
  x4 = x[...,3]
  x5 = x[...,4]
  y1 = (x1+1.7*x2)*torch.sin(x1)  
  y2 = 1.5*x3-0.1*x4*torch.cos(x5+x4-x1)
  return torch.stack((y1, y2),-1)  

def g_sf(x):
    global x_res
    global g_res
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    g0 = (x1+1.7*x2)*numpy.sin(x1)-1.5*x3-0.1*x4*numpy.cos(x5+x4-x1)+0.2*x5**2-x2-1
    if x_res is None:
        x_res = numpy.array([x])
        g_res = numpy.array([g0])
    else:
        x_res = numpy.append(x_res, [x], axis=0)
        g_res = numpy.append(g_res, [g0], axis=0)
    return g0

fun_list = [g]
fun_type = ['gb']

# number of initial points
n_init = 5

# specify probability level (holds for each function)
alpha_list = 0.95

# mapping from full variable space x to input to composite black-box z --> z = A*x
A = torch.zeros(5,5)
A[0,0] = 1.0
A[1,1] = 1.0
A[2,2] = 1.0
A[3,3] = 1.0
A[4,4] = 1.0
# bounds on variables
xL = torch.tensor([-100.0, -100.0,-100.0,-100.0,-100.0])
xU = torch.tensor([100.0, 100.0,100.0, 100.0, 100.0])
bounds = torch.vstack((xL, xU))

# optimal solution
f_opt = -529.87
x_opt = torch.Tensor([98.964258,100,100,99.2243237,-0.25])

print(g(x_opt, h(x_opt)))
print(f_opt)
# create a problem instance
Dolan5D = Problem(g, n_init, h, A, xL, xU, alpha_list, f_opt, x_opt, g_sf)


# In[6]:


### Rosenbrock

# define problem (g0 = obj, g1,g2,... = constraints)
def g(x, y):
  x1 = x[...,0]
  x2 = x[...,1]
  x3 = x[...,2]
  x4 = x[...,3]
  x5 = x[...,4]
  x6 = x[...,5]
  y1 = y[...,0]
  y2 = y[...,1]
  y3 = y[...,2]
  y4 = y[...,3]
  g0 = 100*y1**2 + (1-x1)**2 + \
        100*y2**2 + (1-x2)**2 + \
        100*y3**2 + (1-x3)**2 + \
        100*(x5 - x4**2)**2 + \
        y4 + \
        100*(x6 - x5**2)**2 + \
        (1 - x5)**2
  return -1*g0

# unknown black box function
def h(z):
  z1 = z[...,0]
  z2 = z[...,1]
  z3 = z[...,2]
  z4 = z[...,3]
  y1 = z2 - z1**2
  y2 = z3 - z2**2
  y3 = z4 - z3**2
  y4 = (1-z4)**2
  return torch.stack((y1, y2, y3, y4),-1)

def g_sf(x):
    global x_res
    global g_res
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    g0 = 100*(x2 - x1**2)**2 + (1-x1)**2 + \
        100*(x3 - x2**2)**2 + (1-x2)**2 + \
        100*(x4 - x3**2)**2 + (1-x3)**2 + \
        100*(x5 - x4**2)**2 + \
        (1-x4)**2 + \
        100*(x6 - x5**2)**2 + \
        (1 - x5)**2
    if x_res is None:
        x_res = numpy.array([x])
        g_res = numpy.array([g0])
    else:
        x_res = numpy.append(x_res, [x], axis=0)
        g_res = numpy.append(g_res, [g0], axis=0)
    return g0


fun_list = [g]
fun_type = ['gb']

# number of initial points
n_init = 6

# specify probability level (holds for each function)
alpha_list = 0.95

# mapping from full variable space x to input to composite black-box z --> z = A*x
A = torch.zeros((4,6))
A[0,0] = 1.0
A[1,1] = 1.0
A[2,2] = 1.0
A[3,3] = 1.0

# bounds on variables
xL = torch.tensor([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0])
xU = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
bounds = torch.vstack((xL, xU))

# optimal solution
f_opt = 0.0
x_opt = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# create a problem instance
RB6D = Problem(g, n_init, h, A, xL, xU, alpha_list, f_opt, x_opt, g_sf)


# In[7]:


# Styblinski-Tang function
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
    y1 = y[...,0]
    y2 = y[...,1]
    y3 = y[...,2]
    y4 = y[...,3]
    n=9
    sum = 0
    for i in range(n):
        if i < 4:
            sum = sum + y[...,i]
        else:
            sum = sum + 0.5*(x[...,i]**4 - 16*x[...,i]**2 + 5*x[...,i])
        
    return -1*sum

def h(x):
    x1 = x[...,0]
    x2 = x[...,1]
    x3 = x[...,2]
    x4 = x[...,3]
    y1 = 0.5*(x[...,0]**4 - 16*x[...,0]**2 + 5*x[...,0])
    y2 = 0.5*(x[...,1]**4 - 16*x[...,1]**2 + 5*x[...,1])
    y3 = 0.5*(x[...,2]**4 - 16*x[...,2]**2 + 5*x[...,2])
    y4 = 0.5*(x[...,3]**4 - 16*x[...,3]**2 + 5*x[...,3])
    return torch.stack((y1, y2, y3, y4),-1)

def g_sf(x):
    global x_res
    global g_res
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    x7 = x[6]
    x8 = x[7]
    x9 = x[8]
    n=9
    sum = 0
    for i in range(n):

        sum = sum + 0.5*(x[...,i]**4 - 16*x[...,i]**2 + 5*x[...,i])
    
    g0 = sum
    if x_res is None:
        x_res = numpy.array([x])
        g_res = numpy.array([g0])
    else:
        x_res = numpy.append(x_res, [x], axis=0)
        g_res = numpy.append(g_res, [g0], axis=0)
    return g0


fun_list = [g]
fun_type = ['gb']
# number of initial points
n_init = 9

# specify probability level (holds for each function)
alpha_list = 0.95

# mapping from full variable space x to input to composite black-box z --> z = A*x
A = torch.zeros(4,9)
A[0,0] = 1.0
A[1,1] = 1.0
A[2,2] = 1.0
A[3,3] = 1.0

# bounds on variables
xL = -5.0*torch.ones(9)
xU = 5.0*torch.ones(9)
bounds = torch.vstack((xL, xU))

# optimal solution
f_opt = -39.16599*9
x_opt = -2.903534*torch.ones(1,9)

print(g(x_opt, h(x_opt)))
print(f_opt)


# create a problem instance
Perm9D = Problem(g, n_init, h, A, xL, xU, alpha_list, f_opt, x_opt, g_sf)


# In[8]:


# 7d func Zakhariv Function
def g(x,y):
    x1 = x[...,0]
    x2 = x[...,1]
    x3 = x[...,2]
    x4 = x[...,3]
    x5 = x[...,4]
    x6 = x[...,5]
    x7 = x[...,6]
    y1 = y[...,0]
    n=7
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for i in range(n):

        sum1 = sum1 + x[...,i]**2
        sum2 = sum2 + 0.5*(i+1)*x[...,i]
        sum3 = sum3 + 0.5*(i+1)*x[...,i]
        
    return -1*(sum1 + sum2**2 + (sum3**2)*y1)

def h(x):
    x1 = x[...,0]
    x2 = x[...,1]
    x3 = x[...,2]
    x4 = x[...,3]
    x5 = x[...,4]
    x6 = x[...,5]
    x7 = x[...,6]
    n=7
    sum3=0
    for i in range(n):

        sum3 = sum3 + 0.5*(i+1)*x[...,i]
        
    y = sum3**2
    return y.unsqueeze(-1)
    
def g_sf(x):
    global x_res
    global g_res
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    x7 = x[6]
    n=7
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for i in range(n):

        sum1 = sum1 + x[i]**2
        sum2 = sum2 + 0.5*(i+1)*x[i]
        sum3 = sum3 + 0.5*(i+1)*x[i]
        
    g0 = sum1 + sum2**2 + sum3**4
    if x_res is None:
        x_res = numpy.array([x])
        g_res = numpy.array([g0])
    else:
        x_res = numpy.append(x_res, [x], axis=0)
        g_res = numpy.append(g_res, [g0], axis=0)
    return g0


fun_list = [g]
fun_type = ['gb']
# number of initial points
n_init = 7

# specify probability level (holds for each function)
alpha_list = 0.95

# mapping from full variable space x to input to composite black-box z --> z = A*x
A = torch.eye(7)

# bounds on variables
xL = -5.0*torch.ones(7)
xU = 10.0*torch.ones(7)
bounds = torch.vstack((xL, xU))

# optimal solution
f_opt = 0
x_opt = torch.zeros(1,7)

print(g(x_opt, h(x_opt)))
print(f_opt)


# create a problem instance
Zak7D = Problem(g, n_init, h, A, xL, xU, alpha_list, f_opt, x_opt, g_sf)


# In[9]:


# 8d func Powell (get h(x), fix A)
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
    y4 = y[...,3]
    sum1 = y1 + (x5 + 10*x6)**2
    sum2 = y2 + 5*(x7 - x8)**2
    sum3 = (x2 - 2*x3)**4 + y3
    sum4 = 10*(x1 - x4)**4 + y4
    return -1*(sum1+sum2+sum3+sum4)

def h(x):
    x1 = x[...,0]
    x2 = x[...,1]
    x3 = x[...,2]
    x4 = x[...,3]
    x5 = x[...,4]
    x6 = x[...,5]
    x7 = x[...,6]
    x8 = x[...,7]
    y1 = (x1 + 10*x2)**2
    y2 = 5*(x3 - x4)**2
    y3 = (x6 - 2*x7)**4
    y4 = 10*(x5 - x8)**4
    return torch.stack((y1, y2, y3, y4),-1)
    
def g_sf(x):
    global x_res
    global g_res
    x1 = x[...,0]
    x2 = x[...,1]
    x3 = x[...,2]
    x4 = x[...,3]
    x5 = x[...,4]
    x6 = x[...,5]
    x7 = x[...,6]
    x8 = x[...,7]
    sum1 = (x1 + 10*x2)**2 + (x5 + 10*x6)**2
    sum2 = 5*(x3 - x4)**2 + 5*(x7 - x8)**2
    sum3 = (x2 - 2*x3)**4 + (x6 - 2*x7)**4
    sum4 = 10*(x1 - x4)**4 + 10*(x5 - x8)**4
    g0 = -1*(sum1+sum2+sum3+sum4)
    if x_res is None:
        x_res = numpy.array([x])
        g_res = numpy.array([g0])
    else:
        x_res = numpy.append(x_res, [x], axis=0)
        g_res = numpy.append(g_res, [g0], axis=0)
    return g0

fun_list = [g]
fun_type = ['gb']
# number of initial points
n_init = 8

# specify probability level (holds for each function)
alpha_list = 0.95

# mapping from full variable space x to input to composite black-box z --> z = A*x
A = torch.eye(8)

# bounds on variables
xL = -4.0*torch.ones(8)
xU = 5.0*torch.ones(8)
bounds = torch.vstack((xL, xU))

# optimal solution
f_opt = 0
x_opt = torch.zeros(1,8)

print(g(x_opt, h(x_opt)))
#print(f_opt)


# create a problem instance
Pow8D = Problem(g, n_init, h, A, xL, xU, alpha_list, f_opt, x_opt, g_sf)


# In[10]:


# 5d Friedman func
def g(x,y):
    x1 = x[...,0]
    x2 = x[...,1]
    x3 = x[...,2]
    x4 = x[...,3]
    x5 = x[...,4]
    y1 = y[...,0]
    g0 =  -1*(10*y1 + 20*(x3 - 0.5)**2 + 10*x4 + 5*x5)
    return g0

def h(x):
    x1 = x[...,0]
    x2 = x[...,1]
    y1 = torch.sin(3.14159265*x1*x2) 
    return y1.unsqueeze(-1)

def g_sf(x):
    global x_res
    global g_res
    x1 = x[...,0]
    x2 = x[...,1]
    x3 = x[...,2]
    x4 = x[...,3]
    x5 = x[...,4]
    g0 = 10*numpy.sin(3.14159265*x1*x2) + 20*(x3 - 0.5)**2 + 10*x4 + 5*x5
    g0 =g0
    if x_res is None:
        x_res = numpy.array([x])
        g_res = numpy.array([g0])
    else:
        x_res = numpy.append(x_res, [x], axis=0)
        g_res = numpy.append(g_res, [g0], axis=0)
    return g0

fun_list = [g]
fun_type = ['gb']
# number of initial points
n_init = 5

# specify probability level (holds for each function)
alpha_list = 0.95

# mapping from full variable space x to input to composite black-box z --> z = A*x
A = torch.eye(5)

# bounds on variables
xL = 0*torch.ones(5)
xU = 1*torch.ones(5)
bounds = torch.vstack((xL, xU))

# optimal solution
f_opt = -27.5
x_opt = torch.zeros(1,5)

#print(g_sf(numpy.array(x_opt)))
print(f_opt)


# create a problem instance
Fri5D = Problem(g, n_init, h, A, xL, xU, alpha_list, f_opt, x_opt, g_sf)


# In[11]:


# 3d Wolfe's func 
def g(x,y):
    x1 = x[...,0]
    x2 = x[...,1]
    x3 = x[...,2]
    y1 = y[...,0]
    g0 = 4*(y1)/3 + x3 
    return -1*g0

def h(x):
    x1 = x[...,0]
    x2 = x[...,1]
    y1 = (x1**2 + x2**2 - x1*x2)**0.75 
    return y1.unsqueeze(-1)

def g_sf(x):
    global x_res
    global g_res
    x1 = x[...,0]
    x2 = x[...,1]
    x3 = x[...,2]
    g0 = 4*((x1**2 + x2**2 - x1*x2)**0.75)/3 + x3 
    if x_res is None:
        x_res = numpy.array([x])
        g_res = numpy.array([g0])
    else:
        x_res = numpy.append(x_res, [x], axis=0)
        g_res = numpy.append(g_res, [g0], axis=0)
    return g0

fun_list = [g]
fun_type = ['gb']
# number of initial points
n_init = 3

# specify probability level (holds for each function)
alpha_list = 0.95

# mapping from full variable space x to input to composite black-box z --> z = A*x
A = torch.zeros(2,3)
A[0,0] = 1.0
A[1,1] = 1.0

# bounds on variables
xL = 0*torch.ones(3)
xU = 2*torch.ones(3)
bounds = torch.vstack((xL, xU))

# optimal solution
f_opt = 0
x_opt = torch.ones(1,3)

print(g(x_opt, h(x_opt)))
print(f_opt)

# create a problem instance
Wol3D = Problem(g, n_init, h, A, xL, xU, alpha_list, f_opt, x_opt, g_sf)


# In[25]:


#environmental

### Environmental problem
global g_res
global x_res
g_res = None
x_res = None

def env_cfun(s, t, M, D, L, tau):
  c1 = M / torch.sqrt(4 * math.pi * D * t)
  exp1 = torch.exp(-(s**2) / 4 / D / t)
  term1 = c1 * exp1
  c2 = M / torch.sqrt(4 * math.pi * D * (t - tau))
  exp2 = torch.exp(-((s - L) ** 2) / 4 / D / (t - tau))
  term2 = c2 * exp2
  term2[torch.isnan(term2)] = 0.0
  return term1 + term2

M0 = torch.tensor(10.0)
D0 = torch.tensor(0.07)
L0 = torch.tensor(1.505)
tau0 = torch.tensor(30.1525)

s_size = 3
t_size = 4
if s_size == 3:
    S = torch.tensor([0.0, 1.0, 2.5])
else:
    S = torch.linspace(0.0, 2.5, s_size)
if t_size == 4:
    T = torch.tensor([15.0, 30.0, 45.0, 60.0])
else:
    T = torch.linspace(15.0, 60.0, t_size)
Sgrid, Tgrid = torch.meshgrid(S, T)

# unknown black box function
    # z = [M, D, L, tau]
def h(z):
  z = torch.Tensor(z)
  if z.ndim == 1:
        z = z.unsqueeze(0)
  y = torch.stack([env_cfun(Sgrid, Tgrid, *zi.squeeze()) for zi in z])
  y = y.reshape((z.shape[0],1,-1))
  if len(z.shape) == 2:
    y = y.squeeze(1)
  return y

c_true = env_cfun(Sgrid, Tgrid, M0, D0, L0, tau0)

# define problem (g0 = obj, g1,g2,... = constraints)
def g(x, y):
  # unsqueeze
  if y.shape[-1] == (s_size * t_size):
      y = y.unsqueeze(-1).reshape(*y.shape[:-1], s_size, t_size)
  sq_diffs = (y - c_true).pow(2)
  return sq_diffs.sum(dim=(-1, -2)).mul(-1.0)#.unsqueeze(-1)

def g_sf(x):
    global x_res
    global g_res
    g0 = -1*g(x, h(x))
    if x_res is None:
        x_res = numpy.array([x])
        g_res = numpy.array([g0])
    else:
        x_res = numpy.append(x_res, [x], axis=0)
        g_res = numpy.append(g_res, [g0], axis=0)
    return g0.numpy()

# number of initial points
n_init = 4

# mapping from full variable space x to input to composite black-box z --> z = A*x
A = torch.zeros((4,4))
A[0,0] = 1.0
A[1,1] = 1.0
A[2,2] = 1.0
A[3,3] = 1.0

# bounds on variables
#xL = torch.tensor([7.0, 0.02, 0.01, 30.010])
#xU = torch.tensor([13.0, 0.12, 3.00, 30.295])
#use for direct
xL = torch.tensor([7.0, 0.04, 0.01, 30.010])
xU = torch.tensor([12.0, 0.12, 2.50, 30.295])
bounds = torch.vstack((xL, xU))

# optimal solution
f_opt = 0.0

# create a problem instance
env_nD = Problem(g=g, dim=4, h=h, A=A, xL=xL, xU=xU, alpha=0.95, f_opt=f_opt, x_opt=None, g_sf=g_sf)


# In[13]:


g_sf([0,0,0,0])


# In[14]:


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


# In[15]:


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
        obj = G                                                   # n x b
        #print(obj)
        #print(G)
        improv = (obj - self.best_f.unsqueeze(-1).to(obj)).clamp_min(0)   # n x b

        # create soft constraint function that approximates indicator
        def soft_eval_constraint(rhs, eta):
          return torch.sigmoid(rhs / eta)

        # loop over constraints and multiply by feasibility penalty
        num_cons = G.shape[-1] - 1
        if False:
            for i in range(num_cons):
              cons_eval = soft_eval_constraint(G[...,i+1], self.eta)
              improv = improv.mul(cons_eval)

        # take the average over all samples
        eiccf = improv.mean(dim=0) # b

        return eiccf
    

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


# In[17]:


def optimize_ucq(lcq, bounds, **options):
    cands_nlzd, _ = optimize_acqf(lcq, bounds, **options)
    return cands_nlzd


# In[18]:


from skquant.opt import minimize
import pybobyqa
import cma
import scipy
from botorch.models import HigherOrderGP
from botorch.models.higher_order_gp import FlattenedStandardize
from botorch.models.transforms import Normalize
from linear_operator.settings import _fast_solves
from torch.optim import Adam

def run_main_loop(problem, ninit=2, alpha=0.95, ro=50, n_batches=25, opt_tol=1e-3, seed=0, method='uqb',cons=False, x_data=None, greybox = True, highdim=False):
    batch_size = 1
    torch.manual_seed(seed)
    penalty_weight = ro
    A = problem.A

    bounds = torch.vstack((problem.xL, problem.xU))
    n_ninit = int(ninit*len(problem.xL) + 1)
    
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
        #best_f = G_curr

        # Train model
        if greybox is True:
            if highdim is True:
                model_h = HigherOrderGP(
                                train_Z,
                                train_Y,
                                outcome_transform=FlattenedStandardize(train_Y.shape[1:]),
                                input_transform=Normalize(train_Z.shape[-1]),
                                latent_init="gp")
                          # train model
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(model_h.likelihood, model_h)
                with _fast_solves(True):
                    fit_gpytorch_mll_torch(mll, step_limit=1000, optimizer=partial(Adam, lr=0.01))
                model_h.eval();
            else:
                model_h = train_model(
                          train_Z,
                          train_Y)
        else:
            if highdim is True:
                if G_curr.ndim == 1:
                    G_curr = G_curr.unsqueeze(-1)
                model_h = HigherOrderGP(
                                train_X,
                                G_curr,
                                outcome_transform=FlattenedStandardize(G_curr.shape[1:]),
                                input_transform=Normalize(train_X.shape[-1]),
                                latent_init="gp")
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(model_h.likelihood, model_h)
                with _fast_solves(True):
                    fit_gpytorch_mll_torch(mll, step_limit=1000, optimizer=partial(Adam, lr=0.01))
                model_h.eval();
            else:
                model_h = train_model(
                          train_X,
                          G_curr) 
            
        global x_res
        global g_res
        x_res = None
        g_res = None
        xL = problem.xL
        xU = problem.xU
        # optimize acquisition
        if method == 'uqb':
            ucq = CompositeQuantile(model_h, problem.A, problem.g, problem.alpha)  
        elif method == 'eic':
            incumbent = torch.max(G_curr)
            ucq = botorch.acquisition.analytic.ExpectedImprovement(model_h, incumbent)
        elif method == 'epbo':
            ucq = botorch.acquisition.analytic.UpperConfidenceBound(model_h, beta=4)
        elif method == 'eicf':
            incumbent = torch.max(G_curr)
            ucq = ConstrainedExpectedImprovementComposite(model_h, problem.A, problem.g, incumbent)
        elif method == 'snobfit':
            obj_func = problem.g_sf
            xL = problem.xL
            xU = problem.xU
            bounds = numpy.array([xL.numpy(), xU.numpy()], dtype=float).T
            x0 = numpy.array(x_data[0,:])
            budget=int(5*n_batches)
            result, history = minimize(obj_func, x0, bounds, budget, method='snobfit')
            train_X = torch.Tensor(history[:,1:])
            G_curr = torch.Tensor(history[:,0])
            wall_time = [0]
            break
        elif method == 'cmaes':
            fun = problem.g_sf
            x0 = numpy.array(x_data[0,:])
            #print(x0)
            cma.fmin2(fun, x0, 1,{'bounds':[problem.xL, problem.xU]})
            
            train_X = torch.Tensor(x_res)
            G_curr = torch.Tensor(g_res)
            wall_time = [0]
            break
        elif method == 'bobyqa':
            obj_func = problem.g_sf
            x0 = numpy.array(x_data[0,:])
            bounds = (problem.xL.numpy(), problem.xU.numpy())
            pybobyqa.solve(obj_func, x0, bounds = bounds,  maxfun=int(5*n_batches),rhobeg=0.005)
            train_X = torch.Tensor(x_res)
            G_curr = torch.Tensor(g_res)
            wall_time = [0]
            break
        elif method == 'direct':
            func = problem.g_sf
            bounds = scipy.optimize.Bounds(problem.xL.numpy(), problem.xU.numpy())
            maxfun = n_batches
            scipy.optimize.direct(func=func, bounds=bounds, maxfun=maxfun)
            train_X = torch.Tensor(x_res)
            G_curr = torch.Tensor(g_res)
            wall_time = [0]
            break
            
        try:
            cands, acq_value = optimize_acqf(ucq, bounds=bounds, q=1, num_restarts=3, raw_samples=8192)
            zcands = torch.matmul(cands, A.T)
        except:
            # do random sample
            X_rand = gen_rand_points(bounds, int(2000*problem.dim))
            ucq_val = ucq(X_rand[:,None,:])
            loc = torch.argmax(ucq_val)
            cands = X_rand[loc,:].unsqueeze(0)
            zcands = torch.matmul(cands, A.T)
            print('resorting to random sample')

        # make observations and update data
        if cands.shape[0] > 0:
            train_X = torch.cat([train_X, cands])
            train_Z = torch.cat([train_Z, zcands])
            train_Y = torch.cat([train_Y, problem.h(zcands)])
            G_curr = problem.g(train_X, train_Y)

        if i == 0:
            wall_time = numpy.array([time.monotonic() - tic])
        else:
            wall_time = numpy.append(wall_time, time.monotonic() - tic)
        #print(f"Wall time: {time.monotonic() - tic:1f}")

    return train_X, G_curr, train_Y, wall_time


# In[ ]:


# run for alpha = 0.95, penalty weight in the range of {100, 1000, and 1e4}

problem_list = [Perm9D, Pow8D, Zak7D, RB6D, Col4D, Dolan5D, Ras3D, Booth2D,Wol3D, Fri5D]
solver_list = ['eicf','eic', 'epbo','snobfit', 'cmaes', 'bobyqa', 'direct']
replications = 10
n_batches = 100

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
    for j in range(len(problem_list)):
        for k in range(replications):
            if solver_list[i] == 'uqb':
                X_data, G_data, wall_time = run_main_loop(problem=problem_list[j], ninit=2, alpha=0.95, ro=1e5, n_batches=n_batches, opt_tol=1e-5, seed=(j+9)*(k+9),method=solver_list[i])
                #pd.DataFrame(X_data.numpy()).to_csv(f'Data/Unconstrained/X_problem-{j}_solver-{i}_rep-{k}.csv')
                #pd.DataFrame(G_data.numpy()).to_csv(f'Data/Unconstrained/G_problem-{j}_solver-{i}_rep-{k}.csv')
                #pd.DataFrame(wall_time).to_csv(f'Data/Constrained/time_problem-{j}_solver-{i}_rep-{k}.csv')
                print(f'starting iteration {k+1} and problem {j+1} for solver {i+1}')
            else:
                mc = int(2*problem_list[j].dim + 1)
                x_data = pd.read_csv(f'Data/Unconstrained/X_problem-{j}_solver-0_rep-{k}_alpha-0.75_ninit-1.csv')
                if solver_list[i] == 'eicf':
                    gb = True
                else:
                    gb = False
                x0 = torch.Tensor(numpy.squeeze(x_data[:mc].to_numpy()[:,1:]))
                X_data, G_data, wall_time = run_main_loop(problem=problem_list[j], ninit=2, alpha=0.95, 
                                                          ro=1e5, n_batches=n_batches, opt_tol=1e-5, 
                                                          seed=(j+10)*(k+10),method=solver_list[i],cons=False,
                                                         x_data=x0, greybox = gb)
                if solver_list[i][0] == 'e':
                    R_data = calculate_regret(G_data, maximize=True)
                else:
                    R_data = calculate_regret(-1*G_data, maximize=True)
                pd.DataFrame(R_data).to_csv(f'Data/Unconstrained_solvers/R_problem-{j}_solver-{i}_rep-{k}.csv')
                pd.DataFrame(G_data).to_csv(f'Data/Unconstrained_solvers/G_problem-{j}_solver-{i}_rep-{k}.csv')
                pd.DataFrame(wall_time).to_csv(f'Data/Unconstrained_solvers/time_problem-{j}_solver-{i}_rep-{k}.csv')
                pd.DataFrame(X_data.numpy()).to_csv(f'Data/Unconstrained_solvers/X_problem-{j}_solver-{i}_rep-{k}.csv')
                print(f'finishediteration {k+1} and problem {j+1} for solver {i+1}')


# In[31]:


problem_list = [env_nD]
solver_list = ['uqb','eicf','eic', 'epbo','snobfit', 'cmaes', 'bobyqa', 'direct']
replications = 10
n_batches = 101


for i in range(7,len(solver_list)):
    #i = i+7
    for j in range(len(problem_list)):
        for k in range(replications):
            if solver_list[i] == 'uqb':
                X_data, G_data, Y_data,wall_time = run_main_loop(problem=problem_list[j], ninit=2, alpha=0.95, ro=1e5, n_batches=n_batches, opt_tol=1e-5, seed=(j+10)*(k+10),method=solver_list[i],highdim=True)
                pd.DataFrame(X_data.numpy()).to_csv(f'Data/env_ndim/X_problem-{j}_solver-0_rep-{k}.csv')
                pd.DataFrame(G_data.numpy()).to_csv(f'Data/env_ndim/G_problem-{j}_solver-0_rep-{k}.csv')
                pd.DataFrame(wall_time).to_csv(f'Data/env_ndim/time_problem-{j}_solver-0_rep-{k}.csv')
                print(f'finished iteration {k+1} and problem {j+1} for solver {i+1}')
            else:
                mc = int(2*problem_list[j].dim + 1)
                x_data = pd.read_csv(f'Data/env_ndim/X_problem-{j}_solver-0_rep-{k}.csv')
                if solver_list[i] == 'eicf':
                    gb = True
                else:
                    gb = False
                x0 = torch.Tensor(numpy.squeeze(x_data[:mc].to_numpy()[:,1:]))
                X_data, G_data,Y_data, wall_time = run_main_loop(problem=problem_list[j], ninit=2, alpha=0.95, 
                                                          ro=1e5, n_batches=n_batches, opt_tol=1e-5, 
                                                          seed=(j+10)*(k+10),method=solver_list[i],cons=False,
                                                         x_data=x0, greybox = gb, highdim=True)

                pd.DataFrame(G_data).to_csv(f'Data/env_ndim/G_problem-{j}_solver-{i}_rep-{k}.csv')
                pd.DataFrame(wall_time).to_csv(f'Data/env_ndim/time_problem-{j}_solver-{i}_rep-{k}.csv')
                pd.DataFrame(X_data).to_csv(f'Data/env_ndim/X_problem-{j}_solver-{i}_rep-{k}.csv')
                print(f'finishediteration {k+1} and problem {j+1} for solver {i+1}')

