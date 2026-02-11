# PyTorch
import torch
import torch.nn as nn

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


class TraditionalCELoss(nn.Module):
    def forward(self, logits, labels, params=None, n=None):
        return torch.nn.functional.cross_entropy(logits, labels)


class EDLMSELoss(nn.Module):
    def __init__(self, num_classes, reg_weight=1.0, annealing_steps=10):
        super().__init__()
        self.num_classes = num_classes
        self.reg_weight = reg_weight
        self.annealing_steps = annealing_steps

    def forward(self, logits, labels, params=None, n=None):
        epoch = kwargs.get("epoch", 0)
        evidence = torch.exp(logits)
        alpha = evidence + 1.0
        S = alpha.sum(dim=-1, keepdim=True)

        y = torch.nn.functional.one_hot(targets, self.num_classes).float()
        expected = alpha / S

        err = (y - expected) ** 2
        var = alpha * (S - alpha) / (S ** 2 * (S + 1))
        loss = (err + var).sum(dim=-1)

        annealing = min(1.0, epoch / self.annealing_steps)
        alpha_tilde = y + (1 - y) * alpha
        kl = kl_divergence_dirichlet(alpha_tilde, self.num_classes)
        loss = loss + annealing * self.reg_weight * kl

        return loss.mean()


class EDLCELoss(nn.Module):
    def __init__(self, num_classes, reg_weight=1.0, annealing_steps=10):
        super().__init__()
        self.num_classes = num_classes
        self.reg_weight = reg_weight
        self.annealing_steps = annealing_steps

    def forward(self, logits, labels, params=None, n=None):
        epoch = kwargs.get("epoch", 0)
        evidence = torch.exp(logits)
        alpha = evidence + 1.0
        S = alpha.sum(dim=-1, keepdim=True)

        y = torch.nn.functional.one_hot(targets, self.num_classes).float()

        loss = (y * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1)

        annealing = min(1.0, epoch / self.annealing_steps)
        alpha_tilde = y + (1 - y) * alpha
        kl = kl_divergence_dirichlet(alpha_tilde, self.num_classes)
        loss = loss + annealing * self.reg_weight * kl

        return loss.mean()


def kl_divergence_dirichlet(alpha, num_classes):
    S = alpha.sum(dim=-1, keepdim=True)
    kl = (
        torch.lgamma(S) - torch.lgamma(torch.tensor(num_classes, dtype=alpha.dtype))
        - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
        + ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=-1, keepdim=True)
    )
    return kl.squeeze(-1)