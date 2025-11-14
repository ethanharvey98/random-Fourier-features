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