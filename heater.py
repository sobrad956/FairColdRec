import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from pathlib import Path 
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from data_loader import MovieLensData
from matrix_factor import BiasedMF


@dataclass
class HeaterOutput:
    """Container for model outputs"""
    preds: torch.Tensor
    loss_all: Optional[torch.Tensor] = None
    loss_diff: Optional[torch.Tensor] = None
    reg_loss: Optional[torch.Tensor] = None


class DenseBatchTanh(nn.Module):
    """Dense layer with batch normalization and tanh activation"""
    def __init__(self, in_features: int, out_features: int, do_norm: bool = True):
        super().__init__()
        self.do_norm = do_norm
        self.linear = nn.Linear(in_features, out_features)
        if do_norm:
            self.batch_norm = nn.BatchNorm1d(out_features, momentum=0.1)
        
        # Initialize weights with careful scaling
        nn.init.xavier_normal_(self.linear.weight)
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
        
        l2_loss = torch.sum(self.linear.weight ** 2) + torch.sum(self.linear.bias ** 2)
        return out, l2_loss


class ContentExpertModule(nn.Module):
    """Expert module for content processing"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
            
        self.final = nn.Linear(current_dim, output_dim)
        
        # Initialize weights with Xavier initialization
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.final.weight)
        nn.init.zeros_(self.final.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        reg_loss = 0
        for layer in self.layers:
            x = F.relu(layer(x))  # Using ReLU instead of tanh for deeper networks
            reg_loss += torch.sum(layer.weight ** 2) + torch.sum(layer.bias ** 2)
        
        x = self.final(x)
        reg_loss += torch.sum(self.final.weight ** 2) + torch.sum(self.final.bias ** 2)
        
        return x, reg_loss


class Heater(nn.Module):
    def __init__(self, 
                 latent_rank_in: int,
                 user_content_rank: int,
                 item_content_rank: int,
                 model_select: List[int],
                 rank_out: int,
                 reg: float,
                 alpha: float,
                 dim: int):
        super().__init__()
        self.rank_in = latent_rank_in
        self.phi_u_dim = user_content_rank
        self.phi_v_dim = item_content_rank
        self.model_select = model_select
        self.rank_out = rank_out
        self.reg = reg
        self.alpha = alpha
        self.dim = dim
        
        # Content processing modules
        if self.phi_v_dim > 0:
            self.item_gate = nn.Linear(self.phi_v_dim, dim)
            nn.init.xavier_normal_(self.item_gate.weight)
            nn.init.zeros_(self.item_gate.bias)
            
            self.item_experts = nn.ModuleList([
                ContentExpertModule(self.phi_v_dim, model_select, self.rank_out)
                for _ in range(dim)
            ])
            
        if self.phi_u_dim > 0:
            self.user_gate = nn.Linear(self.phi_u_dim, dim)
            nn.init.xavier_normal_(self.user_gate.weight)
            nn.init.zeros_(self.user_gate.bias)
            
            self.user_experts = nn.ModuleList([
                ContentExpertModule(self.phi_u_dim, model_select, self.rank_out)
                for _ in range(dim)
            ])
        
        # Final transformation layers
        self.user_layer = DenseBatchTanh(self.rank_out, self.rank_out)
        self.item_layer = DenseBatchTanh(self.rank_out, self.rank_out)
        
        # Embedding layers
        self.user_embedding = nn.Linear(self.rank_out, self.rank_out)
        self.item_embedding = nn.Linear(self.rank_out, self.rank_out)
        
        # Initialize embedding weights
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.zeros_(self.user_embedding.bias)
        nn.init.xavier_normal_(self.item_embedding.weight)
        nn.init.zeros_(self.item_embedding.bias)

    def process_content(self, content: torch.Tensor, 
                       gate_layer: nn.Linear,
                       experts: nn.ModuleList) -> Tuple[torch.Tensor, torch.Tensor]:
        # Gate processing with softmax for proper attention
        gate = F.softmax(gate_layer(content), dim=1)
        reg_loss = torch.sum(gate_layer.weight ** 2) + torch.sum(gate_layer.bias ** 2)
        
        # Expert processing
        expert_outputs = []
        for expert in experts:
            output, expert_reg = expert(content)
            expert_outputs.append(output.unsqueeze(1))
            reg_loss += expert_reg
        
        expert_concat = torch.cat(expert_outputs, dim=1)
        
        # Combine gate and experts with proper broadcasting
        output = torch.sum(gate.unsqueeze(2) * expert_concat, dim=1)
        
        return output, reg_loss

    def forward(self, 
                u_in: torch.Tensor,
                v_in: torch.Tensor,
                is_training: bool,
                u_content: Optional[torch.Tensor] = None,
                v_content: Optional[torch.Tensor] = None,
                u_dropout_mask: Optional[torch.Tensor] = None,
                v_dropout_mask: Optional[torch.Tensor] = None) -> HeaterOutput:
        """Forward pass with proper normalization and scaling"""
        reg_loss = 0.0
        
        # Handle missing dropout masks
        if u_dropout_mask is None:
            u_dropout_mask = torch.ones_like(u_in)
        if v_dropout_mask is None:
            v_dropout_mask = torch.ones_like(v_in)
        
        # Process item content if available
        if self.phi_v_dim > 0 and v_content is not None:
            v_content_out, v_content_reg = self.process_content(
                v_content, self.item_gate, self.item_experts)
            reg_loss += v_content_reg
            v_last = v_in * v_dropout_mask + v_content_out * (1 - v_dropout_mask)
            diff_item_loss = self.alpha * torch.sum(torch.square(v_content_out - v_in))
        else:
            v_last = v_in
            diff_item_loss = 0
        
        # Process user content if available
        if self.phi_u_dim > 0 and u_content is not None:
            u_content_out, u_content_reg = self.process_content(
                u_content, self.user_gate, self.user_experts)
            reg_loss += u_content_reg
            u_last = u_in * u_dropout_mask + u_content_out * (1 - u_dropout_mask)
            diff_user_loss = self.alpha * torch.sum(torch.square(u_content_out - u_in))
        else:
            u_last = u_in
            diff_user_loss = 0
        
        # Transform with normalized layers
        u_last, u_reg = self.user_layer(u_last, is_training)
        v_last, v_reg = self.item_layer(v_last, is_training)
        reg_loss += u_reg + v_reg
        
        # Apply final embeddings
        u_embedding = self.user_embedding(u_last)
        v_embedding = self.item_embedding(v_last)
        
        # Normalize embeddings for stable dot product
        u_norm = F.normalize(u_embedding, p=2, dim=1)
        v_norm = F.normalize(v_embedding, p=2, dim=1)
        
        # Calculate predictions with normalized embeddings and scaling
        preds = torch.mm(u_norm, v_norm.t())
        preds = preds * 5.0  # Scale to typical rating range
        
        reg_loss += (torch.sum(self.user_embedding.weight ** 2) + 
                    torch.sum(self.user_embedding.bias ** 2) +
                    torch.sum(self.item_embedding.weight ** 2) + 
                    torch.sum(self.item_embedding.bias ** 2))
        
        reg_loss *= self.reg
        diff_loss = diff_item_loss + diff_user_loss
        
        return HeaterOutput(
            preds=preds,
            loss_all=None,
            loss_diff=diff_loss,
            reg_loss=reg_loss
        )

    def train_step(self, 
                u_in: torch.Tensor,
                v_in: torch.Tensor,
                target: torch.Tensor,
                optimizer: torch.optim.Optimizer,
                u_content: Optional[torch.Tensor] = None,
                v_content: Optional[torch.Tensor] = None,
                u_dropout_mask: Optional[torch.Tensor] = None,
                v_dropout_mask: Optional[torch.Tensor] = None) -> HeaterOutput:
        """Single training step with improved loss calculation"""
        self.train()
        optimizer.zero_grad()
        
        # Forward pass
        output = self(u_in, v_in, True, u_content, v_content, 
                    u_dropout_mask, v_dropout_mask)
        
        # Get predictions for actual items
        preds = torch.diagonal(output.preds)
        
        # Calculate MSE loss with proper scaling
        mse_loss = F.mse_loss(preds, target)
        
        # Total loss with weighted components
        total_loss = mse_loss + output.reg_loss
        if output.loss_diff is not None:
            total_loss = total_loss + output.loss_diff
        
        output.loss_all = total_loss
        
        # Backward pass with gradient clipping
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        
        return output

    def get_recommendations(self, 
                        u_in: torch.Tensor,
                        v_in: torch.Tensor,
                        k: int,
                        train_mat: Optional[torch.sparse.Tensor] = None,
                        u_content: Optional[torch.Tensor] = None,
                        v_content: Optional[torch.Tensor] = None,
                        u_dropout_mask: Optional[torch.Tensor] = None,
                        v_dropout_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get top-k recommendations with proper handling of seen items"""
        self.eval()
        with torch.no_grad():
            output = self(u_in, v_in, False, u_content, v_content,
                        u_dropout_mask, v_dropout_mask)
            scores = output.preds
            
            # Optionally mask out training items
            if train_mat is not None:
                scores = scores - 1e9 * train_mat.to_dense()
            
            # Get top-k items
            _, indices = torch.topk(scores, k)
            
        return indices


class HeaterDataset(Dataset):
    """Dataset for Heater training with improved data handling"""
    def __init__(self, 
                 base_model_embeddings: Tuple[np.ndarray, np.ndarray],
                 ratings: pd.DataFrame,
                 user_content: Optional[np.ndarray] = None,
                 item_content: Optional[np.ndarray] = None):
        self.user_emb, self.item_emb = base_model_embeddings
        self.users = ratings['user_idx'].values
        self.items = ratings['item_idx'].values
        self.ratings = ratings['rating'].values
        self.user_content = user_content
        self.item_content = item_content

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        user_idx = self.users[idx]
        item_idx = self.items[idx]
        
        # Get base model embeddings
        user_emb = self.user_emb[user_idx].astype(np.float32)
        item_emb = self.item_emb[item_idx].astype(np.float32)
        
        # Create dropout masks
        user_mask = np.random.binomial(1, 0.5)
        item_mask = np.random.binomial(1, 0.5)
        
        # Get rating and normalize to [0, 1]
        rating = self.ratings[idx] / 5.0
        
        output = [
            torch.from_numpy(user_emb),
            torch.from_numpy(item_emb),
            torch.tensor(rating, dtype=torch.float32),
            torch.tensor(user_mask, dtype=torch.float32),
            torch.tensor(item_mask, dtype=torch.float32)
        ]
        
        # Add content features if available
        if self.user_content is not None:
            user_content = self.user_content[user_idx].astype(np.float32)
            output.append(torch.from_numpy(user_content))
        if self.item_content is not None:
            item_content = self.item_content[item_idx].astype(np.float32)
            output.append(torch.from_numpy(item_content))
            
        return tuple(output)
    
def train_heater(ml_data: MovieLensData,
                 base_model: BiasedMF,
                 batch_size: int = 1024,
                 num_epochs: int = 50,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Heater:
    """Train Heater model with corrected loss handling"""
    # Get base model embeddings
    user_emb, item_emb = base_model.get_embeddings()
    base_embeddings = (user_emb.float().cpu().numpy(), 
                      item_emb.float().cpu().numpy())
    
    # Create Heater model
    heater = Heater(
        latent_rank_in=base_embeddings[0].shape[1],
        user_content_rank=ml_data.user_content.shape[1] if ml_data.user_content is not None else 0,
        item_content_rank=ml_data.item_content.shape[1] if ml_data.item_content is not None else 0,
        model_select=[100, 50],
        rank_out=base_embeddings[0].shape[1],
        reg=0.0001,
        alpha=1.0,
        dim=8
    ).to(device)
    
    optimizer = torch.optim.AdamW(heater.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    # Create dataset and loader
    heater_train_dataset = HeaterDataset(
        base_embeddings,
        ml_data.train_data,
        ml_data.user_content,
        ml_data.item_content
    )
    
    heater_train_loader = DataLoader(
        heater_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Training loop
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None
    
    print("Starting HEATER training...")
    print(f"Training data size: {len(ml_data.train_data)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Device: {device}")
    
    for epoch in range(num_epochs):
        heater.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(heater_train_loader):
            # Unpack and prepare batch data
            u_emb, i_emb, ratings = [x.float() for x in batch[:3]]
            u_mask, i_mask = [x.float() for x in batch[3:5]]
            u_content = batch[5].float() if len(batch) > 5 else None
            i_content = batch[6].float() if len(batch) > 6 else None
            
            # Move to device
            u_emb = u_emb.to(device)
            i_emb = i_emb.to(device)
            ratings = ratings.to(device)
            u_mask = u_mask.to(device)
            i_mask = i_mask.to(device)
            if u_content is not None:
                u_content = u_content.to(device)
            if i_content is not None:
                i_content = i_content.to(device)
            
            # Training step
            output = heater.train_step(
                u_emb, i_emb, ratings, optimizer,
                u_content=u_content,
                v_content=i_content,
                u_dropout_mask=u_mask.unsqueeze(1),
                v_dropout_mask=i_mask.unsqueeze(1)
            )
            
            # Only track total loss, which already includes all components
            if output.loss_all is not None:
                total_loss += output.loss_all.item()
            
            num_batches += 1
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                current_loss = output.loss_all.item() if output.loss_all is not None else 0
                print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{len(heater_train_loader)} - "
                      f"Current Loss: {current_loss:.4f}")
        
        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        # Step the scheduler
        scheduler.step()
        
        # Print epoch statistics
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in heater.state_dict().items()}
            print("New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after epoch {epoch+1}")
                break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': heater.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, f'heater_checkpoint_epoch_{epoch+1}.pt')
    
    # Load best model state
    if best_model_state is not None:
        heater.load_state_dict(best_model_state)
        print("\nLoaded best model state from training")
    
    print("\nTraining completed!")
    print(f"Best loss achieved: {best_loss:.4f}")
    
    return heater


def save_heater_embeddings(heater: Heater,
                        ml_data: MovieLensData,
                        base_model: BiasedMF,
                        save_path: str,
                        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Save embeddings from trained Heater model with improved handling"""
    heater.eval()
    
    # Convert save_path to Path object and create directory
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        # Get base embeddings and ensure float32
        user_emb, item_emb = base_model.get_embeddings()
        user_emb = user_emb.float()
        item_emb = item_emb.float()
        
        print("Processing user embeddings...")
        # Process users in batches
        user_embeddings = []
        batch_size = 1024
        num_user_batches = (len(user_emb) + batch_size - 1) // batch_size
        
        for i in range(0, len(user_emb), batch_size):
            batch_u = user_emb[i:i+batch_size].to(device)
            
            if ml_data.user_content is not None:
                batch_uc = torch.tensor(
                    ml_data.user_content[i:i+len(batch_u)], 
                    dtype=torch.float32
                ).to(device)
            else:
                batch_uc = None
            
            # Transform user embeddings
            if heater.phi_u_dim > 0 and batch_uc is not None:
                u_content_out, _ = heater.process_content(
                    batch_uc, heater.user_gate, heater.user_experts)
                u_transformed = u_content_out
            else:
                u_transformed = batch_u
            
            u_transformed, _ = heater.user_layer(u_transformed, False)
            u_out = heater.user_embedding(u_transformed)
            user_embeddings.append(u_out.cpu())
            
            if (i // batch_size) % 10 == 0:
                print(f"Processed user batch {i//batch_size + 1}/{num_user_batches}")
        
        print("\nProcessing item embeddings...")
        # Process items in batches
        item_embeddings = []
        num_item_batches = (len(item_emb) + batch_size - 1) // batch_size
        
        for i in range(0, len(item_emb), batch_size):
            batch_i = item_emb[i:i+batch_size].to(device)
            
            if ml_data.item_content is not None:
                batch_ic = torch.tensor(
                    ml_data.item_content[i:i+len(batch_i)], 
                    dtype=torch.float32
                ).to(device)
            else:
                batch_ic = None
            
            # Transform item embeddings
            if heater.phi_v_dim > 0 and batch_ic is not None:
                i_content_out, _ = heater.process_content(
                    batch_ic, heater.item_gate, heater.item_experts)
                i_transformed = i_content_out
            else:
                i_transformed = batch_i
            
            i_transformed, _ = heater.item_layer(i_transformed, False)
            i_out = heater.item_embedding(i_transformed)
            item_embeddings.append(i_out.cpu())
            
            if (i // batch_size) % 10 == 0:
                print(f"Processed item batch {i//batch_size + 1}/{num_item_batches}")
        
        # Concatenate and convert to numpy
        user_embeddings = torch.cat(user_embeddings, dim=0).numpy()
        item_embeddings = torch.cat(item_embeddings, dim=0).numpy()
        
        # Save embeddings
        np.save(save_dir / 'U_emb_Heater.npy', user_embeddings)
        np.save(save_dir / 'I_emb_Heater.npy', item_embeddings)
        print(f"\nSuccessfully saved embeddings to {save_dir}")
    
    