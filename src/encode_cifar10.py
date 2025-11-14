import argparse
import os
import numpy as np
# PyTorch
import torch
import torchvision
# Importing our custom module(s)
import utils

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="encode_cifar10.py")
    parser.add_argument("--batch_size", default=128, help="Batch size (default: 128)", type=int)
    parser.add_argument("--cifar10_dir", default="", help="CIFAR-10 directory (default: \"\")", type=str)
    parser.add_argument("--cifar101_v4_dir", default="", help="CIFAR-10.1 directory (default: \"\")", type=str)
    parser.add_argument("--encoded_path", default="", help="Path to save encoded dataset (default: \"\")", type=str)
    parser.add_argument("--n", default=1000, help="Number of training samples (default: 1000)", type=int)
    parser.add_argument("--num_workers", default=0, help="Number of workers (default: 0)", type=int)
    parser.add_argument("--random_state", default=42, help="Random state (default: 42)", type=int)
    args = parser.parse_args()

    torch.manual_seed(args.random_state)
    
    os.makedirs(os.path.dirname(args.encoded_path), exist_ok=True)
           
    train_dataset, val_dataset, test_dataset = utils.get_cifar10_datasets(args.cifar10_dir, args.n, args.random_state)
    
    data = np.load(f"{args.cifar101_v4_dir}/cifar10.1_v4_data.npy")
    labels = np.load(f"{args.cifar101_v4_dir}/cifar10.1_v4_labels.npy")
    data = torch.from_numpy(data).permute(0, 3, 1, 2).float() / 255.0
    labels = torch.from_numpy(labels).long()
    ood_dataset = utils.TensorDataset(data, labels, test_dataset.transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    ood_dataloader = torch.utils.data.DataLoader(ood_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
                
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    train_metrics = utils.encode_images(model, train_dataloader)
    print(torch.stack(train_metrics["embeddings"]).shape)
    print(torch.stack(train_metrics["labels"]).shape)
        
    val_metrics = utils.encode_images(model, val_dataloader)
    print(torch.stack(val_metrics["embeddings"]).shape)
    print(torch.stack(val_metrics["labels"]).shape)
        
    test_metrics = utils.encode_images(model, test_dataloader)
    print(torch.stack(test_metrics["embeddings"]).shape)
    print(torch.stack(test_metrics["labels"]).shape)
        
    ood_metrics = utils.encode_images(model, ood_dataloader)
    print(torch.stack(ood_metrics["embeddings"]).shape)
    print(torch.stack(ood_metrics["labels"]).shape)
        
    torch.save({
        "X_train": torch.stack(train_metrics["embeddings"]),
        "y_train": torch.stack(train_metrics["labels"]),
        "X_val": torch.stack(val_metrics["embeddings"]),
        "y_val": torch.stack(val_metrics["labels"]),
        "X_test": torch.stack(test_metrics["embeddings"]),
        "y_test": torch.stack(test_metrics["labels"]),
        "X_ood": torch.stack(ood_metrics["embeddings"]),
        "y_ood": torch.stack(ood_metrics["labels"]),
    }, f"{args.encoded_path}")
