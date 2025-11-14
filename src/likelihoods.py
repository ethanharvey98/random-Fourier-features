# PyTorch
import torch
# Importing our custom module(s)
import utils

class GaussianLikelihood(torch.nn.Module):
    def __init__(self, learnable_sigma_y=False, sigma_y=1.0):
        super().__init__()
        
        if learnable_sigma_y:
            self.raw_sigma_y = torch.nn.Parameter(utils.inv_softplus(torch.tensor(sigma_y, dtype=torch.float32)))
        else:
            self.register_buffer("raw_sigma_y", utils.inv_softplus(torch.tensor(sigma_y, dtype=torch.float32)))
                
    def forward(self, logits, labels, reduction):
        device = logits.device
        batch_size = len(logits)
        var = self.noise**2 * torch.ones(size=(batch_size,), device=device)
        return torch.nn.functional.gaussian_nll_loss(logits, labels, var, reduction=reduction, full=True)

    @property
    def sigma_y(self):
        return torch.nn.functional.softplus(self.raw_sigma_y)
    
class BernoulliLikelihood(torch.nn.Module):
    def __init__(self):
        super().__init__()
            
    def forward(self, logits, labels, reduction):
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    
class CategoricalLikelihood(torch.nn.Module):
    def __init__(self):
        super().__init__()
            
    def forward(self, logits, labels, reduction):
        return torch.nn.functional.cross_entropy(logits, labels, reduction=reduction)
    
