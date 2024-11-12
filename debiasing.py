import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR  
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union, Dict
from dataclasses import dataclass
from collections import defaultdict
from data_loader import MovieLensData
from matrix_factor import BiasedMF
from heater import Heater

@dataclass
class ModelOutput:
    """Container for model outputs"""
    preds: torch.Tensor
    loss_all: Optional[torch.Tensor] = None
    loss_r: Optional[torch.Tensor] = None
    reg_loss: Optional[torch.Tensor] = None


class DenseBatchTanh(nn.Module):
    """Dense layer with batch normalization and tanh activation"""
    def __init__(self, in_features: int, out_features: int, do_norm: bool = True):
        super().__init__()
        self.do_norm = do_norm
        self.linear = nn.Linear(in_features, out_features)
        if do_norm:
            self.batch_norm = nn.BatchNorm1d(out_features)
        
        # Initialize weights
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, is_training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.linear(x)
        if self.do_norm:
            if is_training:
                self.batch_norm.train()
            else:
                self.batch_norm.eval()
            out = self.batch_norm(out)
        out = torch.tanh(out)
        
        # Calculate L2 regularization
        l2_loss = torch.sum(self.linear.weight ** 2) + torch.sum(self.linear.bias ** 2)
        
        return out, l2_loss


class RecommendationDataset(Dataset):
    """Dataset for training debiasing model"""
    def __init__(self, R: torch.Tensor, R_output: torch.Tensor, item_indices: np.ndarray):
        self.R = R
        self.R_output = R_output
        self.item_indices = item_indices

    def __len__(self) -> int:
        return len(self.item_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item_idx = self.item_indices[idx]
        return self.R[item_idx], self.R_output[item_idx]


class DebiasingModel(nn.Module):
    """
    Debiasing model using an autoencoder architecture.
    Takes item-user rating vectors, transforms them through hidden layers,
    and reconstructs debiased rating vectors.
    """
    def __init__(self, model_select: List[int], num_user: int, num_item: int, reg: float = 1e-5):
        super().__init__()
        self.reg = reg
        self.num_user = num_user
        self.num_item = num_item
        self.top_k = None
        
        # Build encoder-decoder layers
        self.layers = nn.ModuleList()
        input_dim = num_user
        
        for hidden_dim in model_select:
            self.layers.append(DenseBatchTanh(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        self.output_layer = nn.Linear(input_dim, num_user)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, R_input: torch.Tensor, 
                R_target: Optional[torch.Tensor] = None,
                is_training: bool = True,
                user_input: Optional[torch.Tensor] = None) -> ModelOutput:
        x = R_input
        reg_loss = 0.0
        
        for layer in self.layers:
            x, layer_reg = layer(x, is_training)
            reg_loss += layer_reg
        
        preds = self.output_layer(x)
        reg_loss += torch.sum(self.output_layer.weight ** 2) + torch.sum(self.output_layer.bias ** 2)
        
        if not is_training and user_input is not None:
            preds_t = preds.t()
            user_preds = preds_t[user_input]
            _, top_items = torch.topk(user_preds, k=self.top_k, dim=1)
            return ModelOutput(preds=top_items)

        reg_loss *= self.reg

        if is_training and R_target is not None:
            loss_r = torch.mean(torch.sqrt(torch.sum((preds - R_target) ** 2, dim=1)))
            loss_all = loss_r + reg_loss
            
            return ModelOutput(
                preds=preds,
                loss_all=loss_all,
                loss_r=loss_r,
                reg_loss=reg_loss
            )
        
        return ModelOutput(preds=preds)
    
    def train_step(self, R_input: torch.Tensor, R_target: torch.Tensor, 
                  optimizer: torch.optim.Optimizer) -> ModelOutput:
        """Perform single training step"""
        self.train()
        optimizer.zero_grad()
        
        output = self(R_input, R_target, is_training=True)
        output.loss_all.backward()
        optimizer.step()
        
        return output


def preprocess_ratings(R: torch.Tensor,
                      mask: torch.Tensor,
                      item_warm: np.ndarray,
                      alpha: float) -> torch.Tensor:
    """Preprocess ratings with fixed dimensionality issues"""
    # Print shapes for debugging
    print(f"R shape: {R.shape}")
    print(f"mask shape: {mask.shape}")
    print(f"item_warm shape: {item_warm.shape}, max index: {item_warm.max()}")
    
    # Ensure working with float tensors
    R = R.float()
    mask = mask.float()
    
    # Normalize ratings to [0, 1]
    R_min = R.min()
    R_range = R.max() - R_min
    R = (R - R_min) / R_range if R_range > 0 else R
    R_output = R.clone()
    
    # Calculate position sum and mean
    pos_sum = mask.sum(dim=1, keepdim=True)
    pos_mean = torch.zeros_like(pos_sum)
    
    # Convert item_warm to proper indices
    valid_warm_items = item_warm[item_warm < R.shape[0]]
    print(f"Number of valid warm items: {len(valid_warm_items)}")
    
    # Create warm mask with proper dimensions
    warm_mask = torch.zeros(R.shape[0], dtype=torch.bool, device=R.device)
    warm_mask[valid_warm_items] = True
    warm_mask = warm_mask.unsqueeze(1)
    
    # Calculate means for warm items
    item_mask = warm_mask.expand_as(mask)
    masked_R = R_output * mask
    valid_items = (pos_sum > 0).squeeze(1) & warm_mask.squeeze(1)
    
    if valid_items.any():
        pos_mean[valid_items] = (
            masked_R[valid_items].sum(dim=1, keepdim=True) / 
            pos_sum[valid_items]
        )
    
    # Apply alpha and calculate weights
    pos_mean = pos_mean.pow(alpha)
    weights = torch.zeros_like(pos_sum)
    
    valid_warm = warm_mask.squeeze(1) & (pos_mean.squeeze(1) > 0)
    if valid_warm.any():
        warm_pos_mean = pos_mean[valid_warm]
        max_pos_mean = warm_pos_mean.max()
        if max_pos_mean > 0:
            weights[valid_warm] = max_pos_mean / warm_pos_mean.clamp(min=1e-8)
    
    # Apply weights to ratings
    R_output = R_output * weights * mask + (1 - mask) * R_output
    
    # Rescale back to original range
    R_output = R_output * R_range + R_min
    
    print(f"Output rating matrix shape: {R_output.shape}")
    print(f"Rating range: [{R_output.min():.4f}, {R_output.max():.4f}]")
    
    return R_output

def train_debiasing_model(base_model: Union[BiasedMF, Heater],
                         ml_data: MovieLensData,
                         model_select: List[int] = [100],
                         alpha: float = 4.0,
                         batch_size: int = 50,
                         num_epochs: int = 100,
                         reg: float = 1e-5,
                         device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> DebiasingModel:
    """Train debiasing model with proper dimension handling"""
    
    print("Initializing debiasing model training...")
    print(f"Number of users: {ml_data.n_users}")
    print(f"Number of items: {ml_data.n_items}")
    
    with torch.no_grad():
        if isinstance(base_model, BiasedMF):
            u_emb, i_emb = base_model.get_embeddings()
            R = torch.mm(i_emb, u_emb.t())
        else:  # Heater
            # Get the embeddings after HEATER transformation
            u_emb = base_model.user_embedding.weight.detach()
            i_emb = base_model.item_embedding.weight.detach()
            
            # Print embedding dimensions
            print(f"User embedding shape: {u_emb.shape}")
            print(f"Item embedding shape: {i_emb.shape}")
            
            # Normalize embeddings if they're from HEATER
            u_norm = F.normalize(u_emb, p=2, dim=1)
            i_norm = F.normalize(i_emb, p=2, dim=1)
            
            # Calculate rating matrix
            R = torch.mm(i_norm, u_norm.t()) * 5.0  # Scale back to rating range
            print(f"Generated rating matrix shape: {R.shape}")
    
    # Create mask matrix with proper dimensions
    mask = torch.zeros((R.shape[0], R.shape[1]), device=device)
    print(f"Created mask matrix with shape: {mask.shape}")
    
    # Create rating matrix from training data
    print("Creating rating matrix from training data...")
    valid_entries = 0
    for _, row in ml_data.train_data.iterrows():
        item_idx = row['item_idx']
        user_idx = row['user_idx']
        if item_idx < R.shape[0] and user_idx < R.shape[1]:
            mask[item_idx, user_idx] = 1
            valid_entries += 1
    print(f"Filled mask matrix with {valid_entries} valid entries")
    
    # Move tensors to device
    R = R.to(device)
    mask = mask.to(device)
    
    # Get warm items (items with training data)
    item_warm = ml_data.train_data['item_idx'].unique()
    valid_warm = item_warm[item_warm < R.shape[0]]
    print(f"Number of warm items (before filtering): {len(item_warm)}")
    print(f"Number of valid warm items: {len(valid_warm)}")
    
    # Preprocess ratings
    print("Preprocessing ratings...")
    R_target = preprocess_ratings(R, mask, valid_warm, alpha)
    
    # Create dataset and dataloader
    dataset = RecommendationDataset(R, R_target, valid_warm)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize debiasing model
    print("Initializing debiasing model...")
    debiasing_model = DebiasingModel(
        model_select=model_select,
        num_user=R.shape[1],  # Use actual matrix dimensions
        num_item=R.shape[0],
        reg=reg
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(debiasing_model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.9)
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        total_loss = 0
        total_reg_loss = 0
        total_recon_loss = 0
        num_batches = 0
        
        for R_batch, R_target_batch in train_loader:
            R_batch = R_batch.to(device)
            R_target_batch = R_target_batch.to(device)
            
            output = debiasing_model.train_step(R_batch, R_target_batch, optimizer)
            
            if output.loss_all is not None:
                total_loss += output.loss_all.item()
            if output.reg_loss is not None:
                total_reg_loss += output.reg_loss.item()
            if output.loss_r is not None:
                total_recon_loss += output.loss_r.item()
                
            num_batches += 1
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / num_batches
            avg_reg_loss = total_reg_loss / num_batches
            avg_recon_loss = total_recon_loss / num_batches
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Average Loss: {avg_loss:.4f}')
            print(f'  Average Reg Loss: {avg_reg_loss:.4f}')
            print(f'  Average Recon Loss: {avg_recon_loss:.4f}')
    
    print("\nTraining completed!")
    return debiasing_model