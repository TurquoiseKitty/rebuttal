from scipy import stats
import numpy as np
import torch
from torch.linalg import norm



normalZ = stats.norm(loc=0, scale = 1)

Upper_quant = 0.975
Lower_quant = 0.025

DEFAULT_mean_func = lambda x : 4*np.sin(x/15 * 2 * np.pi)
DEFAULT_hetero_sigma = lambda x : np.clip(0.2 *x *np.abs(np.sin(x)), 0.1, 2)




DEFAULT_layers = [5, 5]


