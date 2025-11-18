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
    
def abstention_metric(logits, labels):
    ds_score = dempster_shafer_score(logits)
    return roc_auc_score(labels, ds_score)