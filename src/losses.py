# PyTorch
import torch

class ERMLoss(torch.nn.Module):
    def __init__(self, likelihood):
        super().__init__()
        self.likelihood = likelihood

    def forward(self, logits, labels, params, n):
        nll = self.likelihood(logits, labels, reduction="mean")
        loss = nll
        return loss

class MAPLoss(torch.nn.Module):
    def __init__(self, likelihood, prior):
        super().__init__()
        self.likelihood = likelihood
        self.prior = prior

    def forward(self, logits, labels, params, n):
        nll = self.likelihood(logits, labels, reduction="mean")
        log_prior = self.prior.log_prob(params)
        loss = nll - (1 / n) * log_prior
        return loss
    
class TemperedIsotropicELBOLoss(torch.nn.Module):
    def __init__(self, model, likelihood, prior, temp):
        super().__init__()
        self.model = model
        self.likelihood = likelihood
        self.prior = prior
        self.temp = temp

    def forward(self, logits, labels, params, n):
        nll = self.likelihood(logits, labels, reduction="mean")
        sigma_q = torch.nn.functional.softplus(self.model.raw_sigma_q)
        kl = self.prior.kl(params, sigma_q**2)
        loss = nll + self.temp * (1 / n) * kl
        return loss
