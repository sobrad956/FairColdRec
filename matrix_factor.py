import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass


@dataclass
class MFOutput:
    """Container for model outputs"""
    preds: torch.Tensor
    loss: Optional[torch.Tensor] = None
    reg_loss: Optional[torch.Tensor] = None


class BiasedMF(nn.Module):
    def __init__(self, 
                 num_users: int,
                 num_items: int,
                 embedding_dim: int = 100,
                 reg: float = 0.0001):
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
                user_ids: torch.Tensor,
                item_ids: torch.Tensor) -> MFOutput:
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
                (user_h**2).sum() +
                (item_h**2).sum() +
                (user_b**2).sum() +
                (item_b**2).sum()
            )
        else:
            reg_loss = None
            
        return MFOutput(preds=preds, reg_loss=reg_loss)

    def train_step(self,
                   user_ids: torch.Tensor,
                   item_ids: torch.Tensor,
                   ratings: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> MFOutput:
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
    
    def get_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get user and item embeddings"""
        return (
            self.user_embeddings.weight.detach(),
            self.item_embeddings.weight.detach()
        )


def train_mf(model: BiasedMF,
             train_loader: torch.utils.data.DataLoader,
             num_epochs: int = 20,
             lr: float = 0.005,
             device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> BiasedMF:
    """
    Train the MF model
    
    Args:
        model: BiasedMF model
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
    
    Returns:
        Trained model
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for user_ids, item_ids, ratings in train_loader:
            # Move to device
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)
            
            # Training step
            output = model.train_step(user_ids, item_ids, ratings, optimizer)
            total_loss += output.loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}')
            
    return model