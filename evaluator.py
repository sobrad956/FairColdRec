import numpy as np
import torch
from sklearn.metrics import ndcg_score
from collections import defaultdict
from tqdm import tqdm


def ndcg_calc_base(model, data_loader, ml_data, k_values=[5, 10, 20, 50], device='cpu'):
    """
    Calculate NDCG at different K values for base MF model
    Args:
        model: BiasedMF model
        data_loader: Test data loader
        ml_data: MovieLens data container
        k_values: List of K values for NDCG calculation
        device: Computing device
    Returns:
        Array of NDCG scores for each K
    """
    # Get all users in test set
    user_ratings = defaultdict(list)
    for batch_triplet in data_loader:
        for i in range(len(batch_triplet[0])):
            user_id, item_id, rating = batch_triplet[0][i], batch_triplet[1][i], batch_triplet[2][i]
            user_ratings[user_id.item()].append((item_id.item(), rating.item()))
    
    model.eval()
    ndcg_scores = []
    all_items = torch.arange(ml_data.n_items, device=device)
    
    with torch.no_grad():
        for k_value in k_values:
            temp_ndcg = []
            for user_id, item_ratings in user_ratings.items():
                # Create ground truth ratings vector (zeros for non-test items)
                true_ratings = np.zeros(ml_data.n_items)
                for item_id, rating in item_ratings:
                    true_ratings[item_id] = rating
                
                # Get predictions for ALL items
                user_tensor = torch.full((ml_data.n_items,), user_id, dtype=torch.long).to(device)
                preds = model(user_tensor, all_items).preds.cpu().numpy()
                
                # Calculate NDCG
                ndcg = ndcg_score(
                    y_true=true_ratings.reshape(1, -1),
                    y_score=preds.reshape(1, -1),
                    k=k_value
                )
                temp_ndcg.append(ndcg)
            
            ndcg_scores.append(np.mean(temp_ndcg))
    
    return ndcg_scores

def ndcg_calc_dropout(base_model,
                    model,
                  test_loader,
                  ml_data,
                  ks = [5, 10, 20, 50],
                  device = 'cuda' if torch.cuda.is_available() else 'cpu') -> float:

    """
    Calculate NDCG@k for dropout model
    
    Args:
        model: Trained dropout model
        test_loader: Test dataloader containing user-item interactions
        base_model: Base MF model for embeddings
        ml_data: MovieLens data container
        k: Cut-off for NDCG calculation
        device: Computing device
        
    Returns:
        Average NDCG@k across users
    """
    model.eval()
    
    # Get all users in test set
    test_users = test_loader.dataset.users.unique()
    
    # Get all items
    all_items = torch.arange(ml_data.n_items, device=device)
    
    ndcg_scores = []
    print("Calculating NDCG per user...")
    
    # Check if model uses content
    has_user_content = hasattr(model, 'phi_u_dim') and model.phi_u_dim > 0
    has_item_content = hasattr(model, 'phi_v_dim') and model.phi_v_dim > 0
    
    # Process each user
    for k_value in ks:
        temp_ndcg = []
        for user_id in tqdm(test_users):
            # Get user's test items and ratings
            user_mask = test_loader.dataset.users == user_id
            user_items = test_loader.dataset.items[user_mask]
            user_ratings = test_loader.dataset.ratings[user_mask]
            
            if len(user_items) < k_value:  # Skip users with too few ratings
                continue
                
            # Create true ratings vector
            true_ratings = torch.zeros(ml_data.n_items, device=device)
            true_ratings[user_items] = user_ratings
            
            # Prepare inputs for all items
            user_emb = base_model.user_embeddings.weight[user_id].expand(ml_data.n_items, -1)
            item_emb = base_model.item_embeddings.weight
            
            # Prepare content if needed
            if has_user_content:
                user_content = torch.tensor(
                    ml_data.user_content[user_id], 
                    dtype=torch.float32,
                    device=device
                ).expand(ml_data.n_items, -1)
            else:
                user_content = None
                
            if has_item_content:
                item_content = torch.tensor(
                    ml_data.item_content,
                    dtype=torch.float32,
                    device=device
                )
            else:
                item_content = None
            
            # Get predictions
            pred_ratings, _, _ = model(
                user_emb,
                item_emb,
                user_content,
                item_content
            )
                    
            # Calculate NDCG
            ndcg = ndcg_score(
                y_true=true_ratings.cpu().numpy().reshape(1, -1),
                y_score=pred_ratings.detach().cpu().numpy().reshape(1, -1),
                k=k_value
            )
            temp_ndcg.append(ndcg)
        ndcg_scores.append(np.mean(temp_ndcg))
        
    return ndcg_scores
