import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass
import numpy as np

#local
from evaluator import ndcg_calc_base, ndcg_calc_sampled


@dataclass
class MFOutput:
    """Container for model outputs"""
    preds: torch.Tensor
    loss: Optional[torch.Tensor] = None
    reg_loss: Optional[torch.Tensor] = None


class BiasedMF(nn.Module):
    def __init__(self, 
                 num_users,
                 num_items,
                 embedding_dim = 100,
                 reg = 0.0001):
        super().__init__()
        self.reg = reg
    
        # User components
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        
        # Item components
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.item_bias = nn.Embedding(num_items, 1)
        
        # Global bias
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        nn.init.normal_(self.user_embeddings.weight, std=0.1)
        nn.init.normal_(self.item_embeddings.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self,
                user_ids,
                item_ids):
        """
        Forward pass of the model
        
        Args:
            user_ids: Tensor of user IDs
            item_ids: Tensor of item IDs
            
        Returns:
            MFOutput containing predictions and optional losses
        """
        # Get embeddings and biases
        user_h = self.user_embeddings(user_ids)
        item_h = self.item_embeddings(item_ids)
        user_b = self.user_bias(user_ids).squeeze()
        item_b = self.item_bias(item_ids).squeeze()
        
        # Compute prediction
        preds = (user_h * item_h).sum(dim=1)  # dot product
        preds = preds + user_b + item_b + self.global_bias
        
        # Calculate regularization if in training
        if self.training:
            reg_loss = self.reg * (
                torch.norm(user_h) + 
                torch.norm(item_h) + 
                torch.norm(user_b) + 
                torch.norm(item_b)
            )
        else:
            reg_loss = None
            
        return MFOutput(preds=preds, reg_loss=reg_loss)

    def train_step(self,
                   user_ids,
                   item_ids,
                   ratings,
                   optimizer):
        """Single training step"""
        self.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = self(user_ids, item_ids)
        
        # Calculate MSE loss
        mse_loss = nn.functional.mse_loss(output.preds, ratings)
        
        # Add regularization
        total_loss = mse_loss + output.reg_loss if output.reg_loss is not None else mse_loss
        output.loss = total_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        return output
    
    def get_embeddings(self):
        """Get user and item embeddings"""
        return (
            self.user_embeddings.weight.detach(),
            self.item_embeddings.weight.detach()
        )
    def get_all_embeddings(self):
        """Get user and item embeddings"""
        return (
            self.user_embeddings.weight.detach(),
            self.user_bias.weight.detach(),
            self.item_embeddings.weight.detach(),
            self.item_bias.weight.detach(),
            self.global_bias,
        )


def train_mf(model,
             train_loader,
             val_loader,
             ml_data,
             num_epochs = 20,
             lr = 0.005,
             device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train the MF model
    
    Args:
        model: BiasedMF model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data,
        ml_data: Movie lens data object
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
    
    Returns:
        Trained model
    """
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=lr,
    #     weight_decay=0.0,  
    #     betas=(0.9, 0.999)
    # )
    train_losses = []
    ndcg_scores = []
    ndcg_test_scores = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for user_ids, item_ids, ratings, _ in train_loader:
            optimizer.zero_grad()
            # Move to device
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)
            
            # Training step
            output = model.train_step(user_ids, item_ids, ratings, optimizer)
            total_loss += output.loss.item()
            num_batches += 1
            
        #Calc NDCG on training set (monitor along with loss for sanity)
        train_ndcg = ndcg_calc_base(model, train_loader, ml_data, k_values=[15,30], device=device)
        ndcg_scores.append(np.mean(train_ndcg))
        
        #Comment this out to speed up, leaving it to check now
        #test_ndcg, test_prec, test_rec = ndcg_calc_sampled(model, val_loader, ml_data, [15,30], device= device)
        #ndcg_test_scores.append(np.mean(test_ndcg))
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        avg_loss = total_loss / num_batches
        #print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f} - Avg Train NDCG: {np.mean(train_ndcg):.4f} - Avg Test NDCG: {np.mean(test_ndcg):.4f}")
        #print(f"Epoch {epoch+1} - Avg Test Prec: {np.mean(test_prec):.4f} - Avg Test Rec: {np.mean(test_rec)}")
            
    return model
