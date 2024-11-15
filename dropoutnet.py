"""https://github.com/layer6ai-labs/DropoutNet"""


import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR  
import scipy
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from matrix_factor import BiasedMF
from data_loader import MovieLensData


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

@torch.no_grad()
def init_weights(net):
    if type(net) == nn.Linear:
        #torch.nn.init.normal_(net.weight, mean=0, std=0.01)
        truncated_normal_(net.weight, std=0.01)
        if net.bias is not None:
            torch.nn.init.constant_(net.bias, 0)


def get_model(latent_rank_in, user_content_rank, item_content_rank, model_select, rank_out):
    model = DeepCF(latent_rank_in, user_content_rank, item_content_rank, model_select, rank_out)
    model.apply(init_weights)
    return model
        


class TanHBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(TanHBlock, self).__init__()
        self.layer = nn.Linear(dim_in, dim_out)
        self.bn = nn.BatchNorm1d(
                num_features=dim_out,
                momentum=0.01,
                eps=0.001
                )

    
    def forward(self, x):
        out = self.layer(x)
        out = self.bn(out)
        out = torch.tanh(out)
        return out

class DeepCF(nn.Module):
    """
    main model class implementing DeepCF
    also stores states for fast candidate generation
    latent_rank_in: rank of preference model input
    user_content_rank: rank of user content input
    item_content_rank: rank of item content input
    model_select: array of number of hidden unit,
        i.e. [200,100] indicate two hidden layer with 200 units followed by 100 units
    rank_out: rank of latent model output
    """

    def __init__(self, latent_rank_in, user_content_rank, item_content_rank, model_select, rank_out):
        super(DeepCF, self).__init__()
        self.rank_in = latent_rank_in
        self.phi_u_dim = user_content_rank
        self.phi_v_dim = item_content_rank
        self.model_select = model_select
        self.rank_out = rank_out

        # inputs
        self.phase = None
        self.target = None
        self.eval_trainR = None
        self.U_pref_tf = None
        self.V_pref_tf = None
        self.rand_target_ui = None

        # outputs in the model
        self.updates = None

        # predictor
        self.tf_topk_vals = None
        self.tf_topk_inds = None
        self.preds_random = None
        self.tf_latent_topk_cold = None
        self.tf_latent_topk_warm = None
        self.eval_preds_warm = None
        self.eval_preds_cold = None
        
        u_dim = self.rank_in + self.phi_u_dim if self.phi_u_dim > 0 else self.rank_in
        v_dim = self.rank_in + self.phi_v_dim if self.phi_v_dim > 0 else self.rank_in

        print ('\tu_concat rank=%s' % str(u_dim))
        print ('\tv_concat rank=%s' % str(v_dim))
        
        u_dims = [u_dim] + self.model_select
        v_dims = [v_dim] + self.model_select
        self.u_layers = nn.ModuleList(TanHBlock(u_dims[i], u_dims[i + 1]) for i in range(len(u_dims) - 1))
        self.v_layers = nn.ModuleList(TanHBlock(v_dims[i], v_dims[i + 1]) for i in range(len(v_dims) - 1))
        
        self.u_emb = nn.Linear(u_dims[-1], self.rank_out)
        self.v_emb = nn.Linear(v_dims[-1], self.rank_out)

    def encode(self, Uin, Vin, Ucontent, Vcontent):
        
        if self.phi_u_dim>0:
            u_concat = torch.cat((Uin, Ucontent), 1)
        else:
            u_concat = Uin

        if self.phi_v_dim>0:
            v_concat = torch.cat((Vin, Vcontent), 1)
        else:
            v_concat = Vin
            
        u_out = u_concat
        for layer in self.u_layers:
            u_out = layer(u_out)
        U_embedding = self.u_emb(u_out)
        
        v_out = v_concat
        for layer in self.v_layers:
            v_out = layer(v_out)
        V_embedding = self.v_emb(v_out)
        return U_embedding, V_embedding
        
    def forward(self, Uin, Vin, Ucontent, Vcontent):
        
        U_embedding, V_embedding = self.encode(Uin, Vin, Ucontent, Vcontent)
        
        preds = U_embedding * V_embedding
        preds = torch.sum(preds, 1)
        return preds, U_embedding, V_embedding

    @torch.no_grad()
    def evaluate(self, recall_k, eval_data, device=None):
        """
        given EvalData runs batch evaluation
        :param recall_k: list of thresholds to compute recall at (information retrieval recall)
        :param eval_data: EvalData instance
        :return: recall array at thresholds matching recall_k
        """
        d = device

        tf_eval_preds_batch = []
        for (batch, (eval_start, eval_stop)) in enumerate(tqdm(eval_data.eval_batch, desc='eval', leave=False)):

            Uin = eval_data.U_pref_test[eval_start:eval_stop, :]
            Vin = eval_data.V_pref_test
            Vcontent = eval_data.V_content_test

            if self.phi_u_dim > 0: 
                Ucontent= eval_data.U_content_test[eval_start:eval_stop, :]
            else:
                Ucontent = None

            Uin = torch.tensor(Uin)
            Vin = torch.tensor(Vin)
            if Ucontent is not None:
                Ucontent = torch.tensor(Ucontent)
            if Vcontent is not None:
                Vcontent = torch.tensor(Vcontent)
            if d is not None:
                Uin = Uin.to(d)
                Vin = Vin.to(d)
                Ucontent = Ucontent.to(d)
                Vcontent = Vcontent.to(d)
            U_embedding, V_embedding = self.encode(Uin, Vin, Ucontent, Vcontent)
            embedding_prod = torch.matmul(U_embedding, V_embedding.t())


            if not eval_data.is_cold:
                eval_trainR = eval_data.tf_eval_train[batch]
                embedding_prod = embedding_prod + eval_trainR

            _, eval_preds = torch.topk(embedding_prod, k=recall_k[-1], sorted=True)
            tf_eval_preds_batch.append(eval_preds.detach().cpu().numpy())


        tf_eval_preds = np.concatenate(tf_eval_preds_batch)

        # filter non-zero targets
        y_nz = [len(x) > 0 for x in eval_data.R_test_inf.rows]
        y_nz = np.arange(len(eval_data.R_test_inf.rows))[y_nz]

        preds_all = tf_eval_preds[y_nz, :]

        recall = []
        for at_k in tqdm(recall_k, desc='recall', leave=False):
            preds_k = preds_all[:, :at_k]
            y = eval_data.R_test_inf[y_nz, :]

            x = scipy.sparse.lil_matrix(y.shape)
            x.data = np.array([z.tolist() for z in np.ones_like(preds_k)]+[[]],dtype=object)[:-1]
            x.rows = np.array([z.tolist() for z in preds_k]+[[]],dtype=object)[:-1]
            z = y.multiply(x)
            recall.append(np.mean(np.divide((np.sum(z, 1)), np.sum(y, 1))))
        return recall
    
    
@dataclass
class DropoutNetOutput:
    """Container for model outputs"""
    preds: torch.Tensor
    loss: Optional[torch.Tensor] = None
    u_embedding: Optional[torch.Tensor] = None
    v_embedding: Optional[torch.Tensor] = None


class DropoutNetDataset(Dataset):
    """Dataset for DropoutNet training"""
    def __init__(self, 
                base_model_embeddings: Tuple[torch.Tensor, torch.Tensor],
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

    def __getitem__(self, idx: int) -> tuple:
        user_idx = self.users[idx]
        item_idx = self.items[idx]
        
        # Get base embeddings
        user_emb = self.user_emb[user_idx]
        item_emb = self.item_emb[item_idx]
        rating = self.ratings[idx]
        
        output = [
            torch.tensor(user_emb, dtype=torch.float),
            torch.tensor(item_emb, dtype=torch.float),
            torch.tensor(rating, dtype=torch.float)
        ]
        
        # Add content if available
        if self.user_content is not None:
            output.append(torch.tensor(self.user_content[user_idx], dtype=torch.float))
        if self.item_content is not None:
            output.append(torch.tensor(self.item_content[item_idx], dtype=torch.float))
            
        return tuple(output)

def train_dropoutnet(ml_data: MovieLensData,
                     base_model: BiasedMF,
                     model_select: List[int] = [800, 400],
                     rank_out: int = 200,
                     dropout_rate: float = 0.5,
                     batch_size: int = 1000,
                     n_scores_per_user: int = 2500,
                     data_batch_size: int = 100,
                     max_data_per_step: int = 2500000,
                     num_epochs: int = 10,
                     learning_rate: float = 0.005,
                     device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> DeepCF:
    """
    Train DropoutNet model for recommendation
    
    Args:
        ml_data: MovieLens data container
        base_model: Trained BiasedMF model
        model_select: Hidden layer dimensions
        rank_out: Output embedding dimension
        dropout_rate: Dropout probability
        batch_size: User batch size
        n_scores_per_user: Number of items to score per user
        data_batch_size: Size of training batches
        max_data_per_step: Maximum training examples per step
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        device: Computing device
        
    Returns:
        Trained DropoutNet model
    """
    print("Initializing DropoutNet training...")
    
    # Get base embeddings
    with torch.no_grad():
        u_emb, i_emb = base_model.get_embeddings()
        u_emb = u_emb.float()
        i_emb = i_emb.float()
        
        # Create expanded embeddings for dropout
        u_emb_expanded = torch.cat([u_emb, torch.zeros(1, u_emb.shape[1])], dim=0)
        i_emb_expanded = torch.cat([i_emb, torch.zeros(1, i_emb.shape[1])], dim=0)
        u_last_idx = u_emb.shape[0]
        i_last_idx = i_emb.shape[0]
    
    # Initialize model
    model = get_model(
        latent_rank_in=u_emb.shape[1],
        user_content_rank=ml_data.user_content.shape[1] if ml_data.user_content is not None else 0,
        item_content_rank=ml_data.item_content.shape[1] if ml_data.item_content is not None else 0,
        model_select=model_select,
        rank_out=rank_out
    ).to(device)
    
    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.9)
    criterion = nn.MSELoss()
    
    # Get all user indices
    user_indices = ml_data.train_data['user_idx'].unique()
    
    print(f"Starting training for {num_epochs} epochs...")
    model.train()
    total_steps = 0
    
    for epoch in range(num_epochs):
        np.random.shuffle(user_indices)
        epoch_loss = 0
        num_batches = 0
        
        # Process users in batches
        for user_batch_start in range(0, len(user_indices), batch_size):
            user_batch = user_indices[user_batch_start:user_batch_start + batch_size]
            
            # Generate targets for the batch
            target_users = np.repeat(user_batch, n_scores_per_user)
            target_users_rand = np.repeat(np.arange(len(user_batch)), n_scores_per_user)
            
            # Get random items for each user
            target_items_rand = [np.random.choice(i_emb.shape[0], n_scores_per_user) 
                               for _ in user_batch]
            target_items_rand = np.array(target_items_rand).flatten()
            
            # Calculate preference scores
            with torch.no_grad():
                batch_u_emb = u_emb[user_batch].to(device)
                preds_pref = torch.mm(batch_u_emb, i_emb.t().to(device))
                target_scores, target_items = torch.topk(preds_pref, k=n_scores_per_user)
                
                target_items = target_items.cpu().numpy()
                target_scores = target_scores.cpu().numpy()
                random_scores = preds_pref.cpu().numpy()[target_users_rand, target_items_rand]
            
            # Combine top-N and random-N items
            target_scores = np.concatenate([target_scores.flatten(), random_scores])
            target_items = np.concatenate([target_items.flatten(), target_items_rand])
            target_users = np.concatenate([target_users, target_users])
            
            # Shuffle and limit data per step
            n_targets = len(target_scores)
            perm = np.random.permutation(n_targets)
            n_targets = min(n_targets, max_data_per_step)
            
            # Process in smaller batches
            for batch_start in range(0, n_targets, data_batch_size):
                batch_end = min(batch_start + data_batch_size, n_targets)
                batch_perm = perm[batch_start:batch_end]
                
                batch_users = target_users[batch_perm]
                batch_items = target_items[batch_perm]
                
                # Apply dropout
                if dropout_rate > 0:
                    n_to_drop = int(np.floor(dropout_rate * len(batch_perm)))
                    drop_user_idx = np.random.permutation(len(batch_perm))[:n_to_drop]
                    drop_item_idx = np.random.permutation(len(batch_perm))[:n_to_drop]
                    
                    batch_u_idx = np.copy(batch_users)
                    batch_i_idx = np.copy(batch_items)
                    
                    batch_u_idx[drop_item_idx] = u_last_idx
                    batch_i_idx[drop_user_idx] = i_last_idx
                else:
                    batch_u_idx = batch_users
                    batch_i_idx = batch_items
                
                # Prepare inputs
                Uin = torch.tensor(u_emb_expanded[batch_u_idx], device=device)
                Vin = torch.tensor(i_emb_expanded[batch_i_idx], device=device)
                
                if ml_data.user_content is not None:
                    Ucontent = torch.tensor(
                        #ml_data.user_content[batch_users].todense(), 
                        ml_data.user_content[batch_users],
                        dtype=torch.float32,
                        device=device
                    )
                else:
                    Ucontent = None
                    
                if ml_data.item_content is not None:
                    Vcontent = torch.tensor(
                        #ml_data.item_content[batch_items].todense(),
                        ml_data.item_content[batch_items],  
                        dtype=torch.float32,
                        device=device
                    )
                else:
                    Vcontent = None
                
                targets = torch.tensor(target_scores[batch_perm], device=device)
                
                # Forward pass
                preds, _, _ = model(Uin, Vin, Ucontent, Vcontent)
                loss = criterion(preds, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                total_steps += 1
                
                if total_steps % 100 == 0:
                    print(f"Step {total_steps}: Loss = {loss.item():.4f}")
        
        # Step the scheduler after each epoch
        scheduler.step()
        
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{num_epochs}: Average Loss = {avg_epoch_loss:.4f}")
    
    print("Training completed!")
    return model



