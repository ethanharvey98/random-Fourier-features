import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# PyTorch
import torch
import torchvision
# Importing our custom module(s)
import layers

def plot_losses(model_history_df, show_figure=False, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(model_history_df["epoch"], model_history_df["train_loss"], label="Train Loss")
    ax.plot(model_history_df["epoch"], model_history_df["val_loss"], label="Val Loss")
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)
    
    if show_figure:
        plt.show()
    
    return fig, ax