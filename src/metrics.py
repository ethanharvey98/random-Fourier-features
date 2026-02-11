from sklearn.metrics import roc_auc_score
# PyTorch
import torch

def balanced_accuracy(preds, labels):
    K = torch.unique(labels)
    return torch.tensor([(preds[labels == k] == k).float().mean() for k in K]).mean()
    
def dempster_shafer_score(logits, trans="exp"):
    N, K = logits.shape
    assert trans in ["exp", "softplus"]
    if trans == "exp":
        evidence = torch.exp(logits)
    elif trans == "softplus":
        evidence = torch.nn.functional.softplus(logits)
    return K / (K + evidence.sum(dim=1))

# Taken directly from Shen et. al
def compute_mutual_information(alphas):
    alpha0 = torch.sum(alphas, 1)
    prob = alphas / alpha0.unsqueeze(-1)
    mi = -torch.sum(
        prob
        * (
            torch.log(prob)
            - torch.digamma(alphas + 1)
            + torch.digamma(alpha0.unsqueeze(-1) + 1)
        ),
        -1,
    )
    return mi

def variance_of_expected(alphas):
    alpha0 = alphas.sum(dim=-1, keepdim=True)
    return (alphas * (alpha0 - alphas) / (alpha0.pow(2) * (alpha0 + 1))).sum(dim=-1)

def abstention_metric(logits, labels):
    ds_score = dempster_shafer_score(logits) # necesitates that the OOD labels are 1 and ID labels are 0
    return roc_auc_score(labels, ds_score)

def subjective_logic_epistemic_uncertainty(logits, trans="exp"):
    ds_score = dempster_shafer_score(logits, trans=trans)
    return ds_score

def mutual_information_epistemic_uncertainty(logits):
    alphas = 1 + torch.exp(logits)
    return compute_mutual_information(alphas)

def variance_based_epistemic_uncertainty(logits):
    alphas = 1 + torch.exp(logits)
    return variance_of_expected(alphas)