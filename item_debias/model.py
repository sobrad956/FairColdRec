import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class ModelOutput:
    """Container for model outputs"""
    preds: torch.Tensor
    loss_all: Optional[torch.Tensor] = None
    loss_r: Optional[torch.Tensor] = None
    reg_loss: Optional[torch.Tensor] = None


class DenseLayer(nn.Module):
    """Dense layer with batch normalization and optional tanh activation"""
    def __init__(self, in_features: int, out_features: int, do_batch_norm: bool = True):
        super().__init__()
        self.do_batch_norm = do_batch_norm
        
        # Linear transformation
        self.linear = nn.Linear(in_features, out_features)
        
        # Batch normalization
        if do_batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_features)
        
        # Initialize weights
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, is_training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        # Linear transform
        out = self.linear(x)
        
        # Apply batch norm if specified
        if self.do_batch_norm:
            if is_training:
                self.batch_norm.train()
            else:
                self.batch_norm.eval()
            out = self.batch_norm(out)
        
        # Apply tanh activation
        out = torch.tanh(out)
        
        # Calculate L2 regularization
        l2_loss = torch.sum(self.linear.weight ** 2) + torch.sum(self.linear.bias ** 2)
        
        return out, l2_loss


class DebiasModel(nn.Module):
    """Main debiasing model architecture"""
    def __init__(self, model_select: List[int], num_user: int, num_item: int, reg: float = 1e-5):
        super().__init__()
        self.reg = reg
        self.num_user = num_user
        self.num_item = num_item
        self.top_k = None  # Will be set during evaluation
        
        # Build model layers
        self.layers = nn.ModuleList()
        input_dim = num_user
        
        # Add hidden layers
        for hidden_dim in model_select:
            self.layers.append(DenseLayer(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        # Add output layer
        self.output_layer = nn.Linear(input_dim, num_user)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, R_input: torch.Tensor, 
                R_output: Optional[torch.Tensor] = None,
                is_training: bool = True,
                user_input: Optional[torch.Tensor] = None) -> ModelOutput:
        """
        Forward pass through the model
        
        Args:
            R_input: Input tensor
            R_output: Target tensor (optional, for training)
            is_training: Whether in training mode
            user_input: User indices for evaluation (optional)
        
        Returns:
            ModelOutput containing predictions and losses or top-k predictions for evaluation
        """
        x = R_input
        reg_loss = 0.0
        
        # Process through hidden layers
        for layer in self.layers:
            x, layer_reg = layer(x, is_training)
            reg_loss += layer_reg
        
        # Output layer
        preds = self.output_layer(x)
        reg_loss += torch.sum(self.output_layer.weight ** 2) + torch.sum(self.output_layer.bias ** 2)
        
        # If in evaluation mode with user input, return top-k predictions
        if not is_training and user_input is not None:
            preds_t = preds.t()  # Transpose to get user dimension first
            user_preds = preds_t[user_input]
            _, top_items = torch.topk(user_preds, k=self.top_k, dim=1)
            return top_items

        # Apply regularization
        reg_loss *= self.reg

        # If in training mode and targets provided, compute losses
        if is_training and R_output is not None:
            # Reconstruction loss (RMSE)
            loss_r = torch.mean(torch.sqrt(torch.sum((preds - R_output) ** 2, dim=1)))
            
            # Total loss
            loss_all = loss_r + reg_loss
            
            return ModelOutput(
                preds=preds,
                loss_all=loss_all,
                loss_r=loss_r,
                reg_loss=reg_loss
            )
        
        return ModelOutput(preds=preds)

    def train_step(self, R_input: torch.Tensor, R_output: torch.Tensor,
                   optimizer: torch.optim.Optimizer) -> ModelOutput:
        """
        Perform a single training step
        
        Args:
            R_input: Input tensor
            R_output: Target tensor
            optimizer: PyTorch optimizer
        
        Returns:
            ModelOutput containing losses and predictions
        """
        self.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = self(R_input, R_output, is_training=True)
        
        # Backward pass
        output.loss_all.backward()
        
        # Update weights
        optimizer.step()
        
        return output


def create_model(model_select: List[int], num_user: int, num_item: int,
                 reg: float = 1e-5) -> DebiasModel:
    """
    Factory function to create model instance
    
    Args:
        model_select: List of hidden layer dimensions
        num_user: Number of users
        num_item: Number of items
        reg: Regularization factor
    
    Returns:
        Initialized DebiasModel
    """
    return DebiasModel(model_select, num_user, num_item, reg)
