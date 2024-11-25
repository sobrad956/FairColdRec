import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR  
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluator import ndcg_calc_base, ndcg_calc_sampled

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from dataclasses import dataclass

from matrix_factor import BiasedMF
from data_loader import MovieLensData
from evaluator import ndcg_calc_dropout, ndcg_calc_dropout_sampled
import torch.optim as optim

# @dataclass
# class DebiasedOutput:
#     """Container for model outputs"""
#     preds: torch.Tensor
#     loss: Optional[torch.Tensor] = None
#     reg_loss: Optional[torch.Tensor] = None

@torch.no_grad()
def weights_init(m):
    if isinstance(m, nn.Linear):  # Apply only to Linear layers
        nn.init.normal_(m.weight, mean=0.0, std=0.02)  # Initialize weights with normal distribution
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # Initialize biases to zero

class Filter(nn.Module):
    def __init__(self, user_embed, embedding_in_dim, item_embedding, user_bias, item_bias, global_bias, batch_size = 1024, embedding_out_dim=100):
        super(Filter, self).__init__()

        self.filtered_user_embeddings = user_embed
        self.item_embedding = item_embedding
        self.user_bias = user_bias
        self.item_bias = item_bias
        self.global_bias = global_bias
        #self.filtered_user_embeddings = nn.Embedding(embedding_in_dim, embedding_out_dim)

        self.filter = nn.Sequential(
            nn.Linear(embedding_out_dim, embedding_out_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(embedding_out_dim, embedding_out_dim),
            nn.BatchNorm1d(embedding_out_dim)
        )
        self.apply(weights_init)
    
    def forward(self, x):
        return self.filter(self.filtered_user_embeddings[x])

    def get_embeddings(self):
        return (
            self.filtered_user_embeddings,
            self.item_embedding,
            self.user_bias,
            self.item_bias,
            self.global_bias
        )

class Discriminator(nn.Module):
    def __init__(self, user_embeddings, user_bias, item_embeddings, item_bias, global_bias, embedding_size=100):
        super(Discriminator, self).__init__()

        self.filtered_user_embeddings, _, _, _, _ = user_embeddings.get_embeddings()
        self.item_embeddings = item_embeddings
        self.user_bias = user_bias
        self.item_bias = item_bias
        self.global_bias = global_bias

        layers = []
        for _ in range(5):  # 5 hidden layers
            layers.append(nn.Linear(embedding_size, embedding_size))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            layers.append(nn.Dropout(p=0.3))
        # Separate outputs: binary prediction and item interaction
        self.hidden_layers = nn.Sequential(*layers)
        self.binary_output = nn.Sequential(
            nn.Linear(embedding_size, 1),
            nn.Sigmoid()  # Binary classification
        )
        self.apply(weights_init)
    
    def forward(self, x, y, user_embeddings):
        self.filtered_user_embeddings, _, _, _, _ = user_embeddings.get_embeddings()
        embed = self.filtered_user_embeddings[x]
        hidden = self.hidden_layers(embed)
        binary_pred = self.binary_output(hidden)  # Predict sensitive class
        item_pred = (embed*self.item_embeddings[y]).sum(dim=1)  # Predict item interactions
        preds = item_pred + self.user_bias[x] + self.item_bias[y] + self.global_bias
        return binary_pred, preds

# Adversarial training setup
def train_adversarial(train_loader, filter_model, discriminator_model, user_embedding, user_bias, item_embeddings, item_bias, global_bias, 
                      epochs=1, filter_lr=1e-3, discriminator_lr=1e-3, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    bce_criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    mse_criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer_filter = optim.Adam(filter_model.parameters(), lr=1e-3)
    optimizer_discriminator = optim.Adam(discriminator_model.parameters(), lr=1e-3)

    train_losses_mse = []
    train_losses_bce = []

    for epoch in range(epochs):
        filter_model = filter_model.to(device)
        discriminator_model = discriminator_model.to(device)
        filter_model.train()
        discriminator_model.train()

        for user_ids, item_ids, ratings, gender in train_loader:
            # Move to device
            user_ids = user_ids.long().to(device)
            item_ids = item_ids.long().to(device)
            ratings = ratings.to(device)
            gender = gender.float().to(device)

            # === Train Discriminator ===
            optimizer_discriminator.zero_grad()
            filtered_embeddings = filter_model(user_ids).detach()  # Stop gradient to filter
            binary_pred, item_pred = discriminator_model(user_ids, item_ids, filter_model)
            
            # Calculate losses
            bce_loss = -bce_criterion(binary_pred.squeeze(), gender)
            mse_loss = mse_criterion(item_pred, ratings)
            discriminator_loss = mse_loss + bce_loss
            discriminator_loss.backward()
            optimizer_discriminator.step()

            # === Train Filter ===
            optimizer_filter.zero_grad()
            filtered_embeddings = filter_model(user_ids)
            binary_pred, _ = discriminator_model(user_ids, item_ids, filter_model) #discriminator_model(filtered_embeddings, user_bias, item_embedding(item_ids), item_bias, global_bias)
            filter_loss = bce_criterion(binary_pred.squeeze(), gender)  # Adversarial: minimize BCE
            filter_loss.backward()
            optimizer_filter.step()
        
        print(f'Epoch [{epoch+1}/{epochs}], Discriminator Loss: {discriminator_loss.item():.4f}, Filter Loss: {filter_loss.item():.4f}')
    return filter_model, discriminator_model


