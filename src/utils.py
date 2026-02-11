import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
# PyTorch
import torch
import torchvision
# Importing our custom module(s)
import layers
import viz_utils

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

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

def train_one_epoch(model, criterion, optimizer, dataloader, num_samples=1, device=torch.device("cpu")):
    model.train()
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        
        batch_size = len(X_batch)
        
        optimizer.zero_grad()
        params = flatten_params(model)
        
        for _ in range(num_samples):
            logits = model(X_batch)
            loss = criterion(logits, y_batch, params, len(dataloader.dataset))
            total_loss += (batch_size / dataset_size) * (1 / num_samples) * loss.item()
            loss.backward()
            
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.mul_(1/num_samples)

        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"], max_norm=1.0)
            
        optimizer.step()
        
    return total_loss
        
def evaluate(model, criterion, dataloader):
    model.eval()
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    
    with torch.no_grad():
        
        total_loss = 0.0
        for X_batch, y_batch in dataloader:

            batch_size = len(X_batch)
            
            params = flatten_params(model)
            logits = model(X_batch)
            loss = criterion(logits, y_batch, params, len(dataloader.dataset))
            total_loss += (batch_size / dataset_size) * loss.item()
            
    return total_loss

def save_run(result_dir, args, model, likelihood, prior, model_history_df,
             train_results, val_results, test_results, ood_results):
    os.makedirs(result_dir, exist_ok=True)

    viz_utils.plot_losses(model_history_df, save_path=f"{args.result_dir}/train_val_loss.png")
    
    # Save model state dicts
    if likelihood is not None and prior is not None:
        torch.save({
            "model": model.state_dict(),
            "likelihood": likelihood.state_dict(),
            "prior": prior.state_dict(),
        }, f"{result_dir}/model.pth")
    else:
        torch.save({
            "model": model.state_dict(),
        }, f"{result_dir}/model.pth")
    
    # Save training history
    model_history_df.to_csv(f"{result_dir}/training_history.csv", index=False)
    
    # Save predictions with LABELS
    torch.save({
        "train_probs": train_results[0],
        "train_labels": train_results[5],
        "val_probs": val_results[0],
        "val_labels": val_results[5],
        "test_probs": test_results[0],
        "test_labels": test_results[5],
        "ood_probs": ood_results[0],
        "ood_labels": ood_results[5],
        "train_logits": train_results[4],
        "val_logits": val_results[4],
        "test_logits": test_results[4],
        "ood_logits": ood_results[4],
    }, f"{result_dir}/predictions.pth")
    
    # Save metrics summary
    metrics_summary = {
        "args": vars(args),
        "train": {
            "accuracy": train_results[1],
            "balanced_accuracy": train_results[2],
            "per_class_accuracy": train_results[3],
        },
        "val": {
            "accuracy": val_results[1],
            "balanced_accuracy": val_results[2],
            "per_class_accuracy": val_results[3],
        },
        "test": {
            "accuracy": test_results[1],
            "balanced_accuracy": test_results[2],
            "per_class_accuracy": test_results[3],
        },
        "ood": {
            "accuracy": ood_results[1],
            "balanced_accuracy": ood_results[2],
            "per_class_accuracy": ood_results[3],
        },
    }
    
    with open(f"{result_dir}/metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    
    print(f"Run saved to {result_dir}")

def load_cifar_data(repo_dir, n=100, random_state=1001, batch_size=128, device=torch.device("cpu")):
    print(f"Loading CIFAR-10: \n -n={n}\n  -random_state={random_state}")
    datasets = torch.load(f"{repo_dir}/datasets/CIFAR-10/n={n}_random_state={random_state}.pth", map_location=device)

    full_dataset = torch.utils.data.TensorDataset(
        torch.cat([datasets["X_train"], datasets["X_val"]], dim=0),
        torch.cat([datasets["y_train"], datasets["y_val"]], dim=0),
    )
    train_dataset = torch.utils.data.TensorDataset(datasets["X_train"], datasets["y_train"])
    val_dataset = torch.utils.data.TensorDataset(datasets["X_val"], datasets["y_val"])
    test_dataset = torch.utils.data.TensorDataset(datasets["X_test"], datasets["y_test"])
    ood_dataset = torch.utils.data.TensorDataset(datasets["X_ood"], datasets["y_ood"])

    full_dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    ood_dataloader = torch.utils.data.DataLoader(ood_dataset, batch_size=batch_size)

    return full_dataloader, train_dataloader, val_dataloader, test_dataloader, ood_dataloader, full_dataset, train_dataset, val_dataset, test_dataset, ood_dataset

def get_predictions_and_accuracy(dataloader, model, num_samples=10_000, device=torch.device("cpu")):
    model.eval()
    model.to(device)

    all_probs = []
    all_logits = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            if num_samples is not None:
                probs = model.predict_proba(X_batch, num_samples=num_samples)
                logits = model.predict_logits(X_batch, num_samples=num_samples)
            else:
                probs = model.predict_proba(X_batch)
                logits = model(X_batch)
            preds = probs.argmax(dim=-1)
            
            all_probs.append(probs)
            all_logits.append(logits)
            all_preds.append(preds)
            all_labels.append(y_batch)

    all_probs = torch.cat(all_probs, dim=0)
    all_logits = torch.cat(all_logits, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    accuracy = (all_preds == all_labels).float().mean().item()

    unique_classes = torch.unique(all_labels)
    per_class_accuracy = {}
    
    for cls in unique_classes:
        cls_mask = all_labels == cls
        cls_correct = (all_preds[cls_mask] == all_labels[cls_mask]).float().mean().item()
        per_class_accuracy[cls.item()] = cls_correct

    balanced_accuracy = sum(per_class_accuracy.values()) / len(per_class_accuracy)

    return all_probs, accuracy, balanced_accuracy, per_class_accuracy, all_logits, all_labels 