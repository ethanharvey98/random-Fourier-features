import os
import json
import tqdm
import types
import argparse
import itertools
import pandas as pd
import matplotlib.pyplot as plt
#PyTorch
import torch

#OurCode
import sys
sys.path.append("../../../src/")
import layers
import likelihoods
import losses
import metrics
import priors
import utils

# Parsing arguments for experiment
parser = argparse.ArgumentParser(description="Trains an RFF + Laplace model and saves the model and training information")

parser.add_argument("--repo_dir", type=str, default="../../..", 
                    help="Path to the repository root directory")
parser.add_argument("--result_dir", type=str, default="../../../results/test",
                    help="Path to result directory")
parser.add_argument("--n", type=int, default=100, 
                    help="Number of training samples")
parser.add_argument("--random_state", type=int, default=1001, 
                    help="Random seed for reproducibility")
parser.add_argument("--batch_size", type=int, default=128, 
                    help="Number of samples per training batch")
parser.add_argument("--rank", type=int, default=1024,
                    help="Number of random Fourier features")
parser.add_argument("--lengthscale", type=float, default=20.0,
                    help="Kernel lengthscale parameter")
parser.add_argument("--outputscale", type=float, default=1.0,
                    help="Kernel outputscale parameter")
parser.add_argument("--prediction_samples", type=int, default=10_000,
                    help="Number of samples the model will take when making predictions")
parser.add_argument("--epochs", type=int, default=10_000, 
                    help="Number of epochs to train")
parser.add_argument("--learning_rate", type=float, default=0.01,
                    help="Learning rate for optimizer")
parser.add_argument("--patience", type=int, default=50, 
                    help="Number of epochs to train")


def main():
    # Load in args
    args = parser.parse_args()

    # Checking device we are working on
    device = utils.get_device()
    print(f"Using device: {device}")

    # Loading CIFAR10 and CIFAR10.1
    print(f"Loading dataloader of size n={args.n} with random_state={args.random_state}")
    full_dataloader, train_dataloader, val_dataloader, test_dataloader, ood_dataloader, *_ = utils.load_cifar_data(repo_dir=args.repo_dir, n=args.n, random_state=args.random_state, batch_size=args.batch_size, device=device)

    # Training model on train and early stopping on validation
    model = layers.RFFLaplace(in_features=2048, out_features=10, rank=args.rank, lengthscale=args.lengthscale, outputscale=args.outputscale)
    likelihood = likelihoods.CategoricalLikelihood()
    prior = priors.GaussianPrior(learnable_tau=False, tau=1.0)
    model.to(device)

    map_objective = losses.MAPLoss(likelihood, prior)
    cross_entropy = losses.ERMLoss(likelihood)
    optimizer = torch.optim.Adam([{"params": model.parameters()}, {"params": likelihood.parameters()}, {"params": prior.parameters()}], lr=args.learning_rate, weight_decay=0.0)

    state_dict = {
        "model": model.state_dict(),
        "likelihood": likelihood.state_dict(),
        "prior": prior.state_dict(),
    }

    columns = ["lr", "epoch", "train_loss", "val_loss"]
    model_history_df = pd.DataFrame(columns=columns)
    best_state_dict = None
    early_stopping_count = 0
    
    for epoch in tqdm.tqdm(range(args.epochs)):
        train_loss = utils.train_one_epoch(model, map_objective, optimizer, train_dataloader)
        val_loss = utils.evaluate(model, cross_entropy, val_dataloader)
        
        model_history_df.loc[len(model_history_df)] = [args.learning_rate, epoch, train_loss, val_loss]
        
        if val_loss == model_history_df["val_loss"].min(): 
            best_state_dict = {
                "model": model.state_dict(),
                "likelihood": likelihood.state_dict(),
                "prior": prior.state_dict(),
            }
        else:
            if args.patience <= early_stopping_count:
                break
            early_stopping_count += 1
        
    model.load_state_dict(best_state_dict["model"])
    likelihood.load_state_dict(best_state_dict["likelihood"])
    prior.load_state_dict(best_state_dict["prior"])

    # Updating Covariance for predictions
    model.eval()
    model.update_covariance_from_dataloader(train_dataloader, device=device)

    # Evaluating model generally
    train_results = utils.get_predictions_and_accuracy(
        train_dataloader, model, num_samples=args.prediction_samples, device=device)
    val_results = utils.get_predictions_and_accuracy(
        val_dataloader, model, num_samples=args.prediction_samples, device=device)
    test_results = utils.get_predictions_and_accuracy(
        test_dataloader, model, num_samples=args.prediction_samples, device=device)
    ood_results = utils.get_predictions_and_accuracy(
        ood_dataloader, model, num_samples=args.prediction_samples, device=device)

    utils.save_run(
        result_dir=args.result_dir,
        args=args,
        model=model,
        likelihood=likelihood,
        prior=prior,
        model_history_df=model_history_df,
        train_results=train_results,
        val_results=val_results,
        test_results=test_results,
        ood_results=ood_results,
    )

if __name__ == "__main__":
    main()