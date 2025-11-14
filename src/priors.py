import math
# PyTorch
import torch
# Importing our custom module(s)
import utils

class GaussianPrior(torch.nn.Module):
    def __init__(self, learnable_tau=False, tau=1.0):
        super().__init__()
        
        if learnable_tau:
            self.raw_tau = torch.nn.Parameter(utils.inv_softplus(torch.tensor(tau, dtype=torch.float32)))
        else:
            self.register_buffer("raw_tau", utils.inv_softplus(torch.tensor(tau, dtype=torch.float32)))
        
    @property
    def tau(self):
        return torch.nn.functional.softplus(self.raw_tau)

    def log_prob(self, params):
        params_diff_norm = (params**2).sum()
        log_norm_const = len(params) * math.log(2.0 * math.pi)
        log_det = len(params) * torch.log(self.tau)
        quad_term = (1 / self.tau) * params_diff_norm
        log_prob = -0.5 * (log_norm_const + log_det + quad_term)
        return log_prob
        