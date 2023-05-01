### Toy Hydrology

from quantile_bound import cuqb
import torch

# define problem (g0 = obj, g1,g2,... = constraints)
def g0(x):
  x1 = x[...,0]
  x2 = x[...,1]
  return x1 + x2

def g1(x, y):
  x1 = x[...,0]
  x2 = x[...,1]
  y1 = y[...,0]
  return (1.5 - x1 - 2.0*x2 - 0.5*torch.sin(-4.0*3.141519*x2 + y1))

def g2(x):  
  x1 = x[...,0]
  x2 = x[...,1]
  return (x1**2 + x2**2 - 1.5)

# unknown black box function
def h(z):
  z1 = z[...,0]
  y1 = 2*3.141519*z1**2
  y1 = y1.unsqueeze(-1)
  return y1

# specify function list and type
# grey-box functions are all assumed to be of the form g(x, y) where y = h(x) is the composite black-box
fun_list = [g0, g1, g2]
fun_type = ["wb", "gb", "wb"]

# specify probability level for each function
alpha_list = [None, 0.05, None]

# mapping from full variable space x to input to composite black-box z --> z = A*x
A = torch.zeros((1,2))
A[0,0] = 1.0

# bounds on variables
xL = torch.tensor([0.0, 0.0])
xU = torch.tensor([1.0, 1.0])

# optimal solution (as list)
f_opt = 0.5998
x_opt = [0.1951, 0.4047]

# create a problem instance
Toy_Hydrology = cuqb.Problem(fun_list, fun_type, h, A, xL, xU, alpha_list, f_opt, x_opt)

# run problem
res = cuqb.Configuration(Toy_Hydrology, acq_f = None, Ninit=5, NFE=25, x_data=None, z_data=None, h_data=None, fun_data=None,\
                    n_samp_quantile=1000, M=1e5, eps=1e-4, opt_method="ipopt", raw_samples=1000, num_multi_starts=1, seed=1)
res.run_main_loop()
print(res.get_current_best_point())