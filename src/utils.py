import os
import numpy as np
from sklearn.model_selection import train_test_split
# PyTorch
import torch
import torchvision
# Importing our custom module(s)
import layers

def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))

def worker_init_fn(worker_id):
    # This worker initialization function sets CPU affinity for each worker to 
    # all available CPUs, significantly improving GPU utilization when using 
    # num_workers > 0 (see https://github.com/pytorch/pytorch/issues/99625).
    os.sched_setaffinity(0, range(os.cpu_count()))
    
def get_mean_and_std(dataset, indices, dims=(1, 2)):
    
    means, stds = [], []

    for image, label in map(dataset.__getitem__, indices):
        means.append(torch.mean(image, dim=dims).tolist())
        stds.append(torch.std(image, dim=dims).tolist())

    return torch.tensor(means).mean(dim=0), torch.tensor(stds).mean(dim=0)

class TensorDataset(torch.utils.data.Dataset):
    
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.transform:
            return (self.transform(self.X[index]), self.y[index])
        else:
            return (self.X[index], self.y[index])

class TensorSubset(torch.utils.data.Dataset):
    
    def __init__(self, dataset, indices, transform=None):
        X, y = zip(*[dataset[i] for i in indices])
        self.X = torch.stack(X)
        self.y = torch.tensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.transform:
            return (self.transform(self.X[index]), self.y[index])
        else:
            return (self.X[index], self.y[index])
    
def get_cifar10_datasets(root, n, random_state):

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=root, 
        train=True, 
        transform=transform, 
        download=True
    )
    full_test_dataset = torchvision.datasets.CIFAR10(
        root=root, 
        train=False, 
        transform=transform, 
        download=True
    )

    if n == len(full_train_dataset):
        train_and_val_indices = np.arange(0, len(full_train_dataset))
    else:
        train_and_val_indices, _ = train_test_split(
            np.arange(0, len(full_train_dataset)), 
            test_size=None, 
            train_size=n, 
            random_state=random_state, 
            shuffle=True, 
            stratify=np.array(full_train_dataset.targets),
        )
        
    val_size = int((1/5) * n)
    train_indices, val_indices = train_test_split(
        train_and_val_indices, 
        test_size=val_size, 
        train_size=n-val_size, 
        random_state=random_state, 
        shuffle=True, 
        stratify=np.array(full_train_dataset.targets)[train_and_val_indices],
    )

    mean, std = get_mean_and_std(full_train_dataset, train_indices)
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.Resize(size=(224, 224)),
    ])

    train_dataset = TensorSubset(full_train_dataset, train_indices, transform)
    val_dataset = TensorSubset(full_train_dataset, val_indices, transform)
    test_dataset = TensorSubset(full_test_dataset, range(len(full_test_dataset)), transform)
        
    return train_dataset, val_dataset, test_dataset

def add_variational_layers(module, raw_sigma):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, layers.VariationalLinear(child, raw_sigma))
        elif isinstance(child, torch.nn.Conv2d):
            setattr(module, name, layers.VariationalConv2d(child, raw_sigma))
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(module, name, layers.VariationalBatchNorm2d(child, raw_sigma))
        else:
            add_variational_layers(child, raw_sigma)
            
def use_posterior(self, flag):
    for child in self.modules():
        if isinstance(child, (
            layers.VariationalLinear, 
            layers.VariationalConv2d, 
            layers.VariationalBatchNorm2d,
        )):
            child.use_posterior = flag

def encode_images(model, dataloader):
    
    device = torch.device("cuda:0" if next(model.parameters()).is_cuda else "cpu")
    model.eval()   
    
    metrics = {"embeddings": [], "labels": []}
            
    with torch.no_grad():
        
        for images, labels in dataloader:
                        
            if device.type == "cuda":
                images, labels = images.to(device), labels.to(device)
                
            embeddings = model(images)
            
            if device.type == "cuda":
                labels, embeddings = labels.cpu(), embeddings.cpu()
                
            metrics["labels"].extend(labels)
            metrics["embeddings"].extend(embeddings)
            
    return metrics

def flatten_params(model, excluded_params=["raw_lengthscale", "raw_outputscale", "raw_sigma_q", "raw_sigma_y", "raw_tau"]):
    return torch.cat([param.view(-1) for name, param in model.named_parameters() if param.requires_grad and name not in excluded_params])


