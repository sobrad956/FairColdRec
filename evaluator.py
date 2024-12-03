import numpy as np
import torch
from sklearn.metrics import ndcg_score
from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import List
from collections import defaultdict
from tqdm import tqdm

from data_loader import MovieLensDataset

def analyze_mdg_percentiles(mdg_scores, percentiles = [10, 20, 90]):

    # Convert to array and sort
    items = np.array(list(mdg_scores.keys()))
    scores = np.array(list(mdg_scores.values()))
    sort_idx = np.argsort(scores)
    sorted_scores = scores[sort_idx]
    sorted_items = items[sort_idx]
    
    results = {}
    n_items = len(scores)
    
    # Calculate for bottom percentiles
    for p in sorted(percentiles):
        if p < 50:  # Bottom percentile
            n_bottom = int(np.ceil(n_items * (p/100)))
            bottom_scores = sorted_scores[:n_bottom]
            results[f'bottom_{p}'] = {
                'mean': np.mean(bottom_scores),
                'std': np.std(bottom_scores),
                'n_items': n_bottom,
                'items': sorted_items[:n_bottom]
            }
        else:  # Top percentile
            n_top = int(np.ceil(n_items * ((100-p)/100)))
            top_scores = sorted_scores[-n_top:]
            results[f'top_{100-p}'] = {
                'mean': np.mean(top_scores),
                'std': np.std(top_scores),
                'n_items': n_top,
                'items': sorted_items[-n_top:]
            }
    
    # Overall stats
    results['all'] = {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'n_items': n_items
    }
    
    return results

def mdg_calc_base(model, data_loader, ml_data, k=100, device='cpu'):
    """Calculate MDG using only test set items"""
    user_ratings = defaultdict(list)
    item_users = defaultdict(list)
    mdg_scores = defaultdict(list)
    
    # Group ratings by user
    for batch_triplet in data_loader:
        for i in range(len(batch_triplet[0])):
            user_id, item_id, rating = batch_triplet[0][i], batch_triplet[1][i], batch_triplet[2][i]
            user_ratings[user_id.item()].append((item_id.item(), rating.item()))
            if rating.item() > 0:
                item_users[item_id.item()].append(user_id.item())
    
    model.eval()
    
    with torch.no_grad():
        for user_id, item_ratings in user_ratings.items():
            # Get only test items for this user
            test_items = torch.tensor([item[0] for item in item_ratings]).to(device)
            user_tensor = torch.full((len(test_items),), user_id, dtype=torch.long).to(device)
            
            # Predict only test items
            preds = model(user_tensor, test_items).preds.cpu().numpy()
            
            # Calculate rankings
            sorted_indices = np.argsort(-preds)
            rankings = {test_items[idx].item(): rank + 1 for rank, idx in enumerate(sorted_indices)}
            
            # Update MDG scores
            for item_id, rating in item_ratings:
                if rating > 0:
                    rank = rankings[item_id]
                    if rank <= k:
                        mdg_scores[item_id].append(1.0 / np.log2(1 + rank))
                    else:
                        mdg_scores[item_id].append(0)
    
    # Calculate final MDG scores
    final_mdg = {}
    for item_id, gains in mdg_scores.items():
        matched_users = len(item_users[item_id])
        if matched_users > 0:
            final_mdg[item_id] = sum(gains) / matched_users
    
    mdg_analysis = analyze_mdg_percentiles(final_mdg)
    return final_mdg, mdg_analysis

def mdg_calc_dropout(model, 
                    base_model,
                    data_loader, 
                    ml_data,
                    k=100,
                    total_eval_items=200, 
                    device='cpu'):
    """Calculate MDG for dropout model using test items and sampled negatives"""
    user_ratings = defaultdict(list)
    item_users = defaultdict(list)
    mdg_scores = defaultdict(list)
    all_items = set(range(ml_data.n_items))
    
    # Group test ratings
    for batch_triplet in data_loader:
        for i in range(len(batch_triplet[0])):
            user_id = batch_triplet[0][i].item()
            item_id = batch_triplet[1][i].item()
            rating = batch_triplet[2][i].item()
            user_ratings[user_id].append((item_id, rating))
            if rating > 0:
                item_users[item_id].append(user_id)
    
    model.eval()
    base_model.eval()
    
    # Get base embeddings once
    with torch.no_grad():
        u_emb, i_emb = base_model.get_embeddings()
        u_emb = u_emb.to(device)
        i_emb = i_emb.to(device)
        
        # Prepare content features if available
        u_content = None
        i_content = None
        if ml_data.user_content is not None:
            u_content = torch.tensor(ml_data.user_content, dtype=torch.float32).to(device)
        if ml_data.item_content is not None:
            i_content = torch.tensor(ml_data.item_content, dtype=torch.float32).to(device)
        
        for user_id, item_ratings in user_ratings.items():
            # Get test items and sample negatives
            test_items = set(item[0] for item in item_ratings)
            n_test = len(test_items)
            n_neg = max(0, total_eval_items - n_test)
            
            available_neg = list(all_items - test_items)
            if n_neg > 0 and available_neg:
                n_neg = min(n_neg, len(available_neg))
                neg_items = set(np.random.choice(available_neg, n_neg, replace=False))
            else:
                neg_items = set()
            
            eval_items = list(test_items | neg_items)
            
            # Get user embedding and encode
            user_emb = u_emb[user_id:user_id+1]
            items_emb = i_emb[eval_items]
            
            u_encoded, i_encoded = model.encode(
                user_emb,
                items_emb,
                u_content[user_id:user_id+1] if u_content is not None else None,
                i_content[eval_items] if i_content is not None else None
            )
            
            # Calculate predictions with bias terms
            dot_products = torch.mm(i_encoded, u_encoded.t()).squeeze()
            user_bias = base_model.user_bias(torch.tensor([user_id], device=device)).squeeze()
            item_biases = base_model.item_bias(torch.tensor(eval_items, device=device)).squeeze()
            pred_ratings = dot_products + user_bias + item_biases + base_model.global_bias
            
            # Get rankings
            sorted_indices = torch.argsort(pred_ratings, descending=True).cpu().numpy()
            rankings = {eval_items[idx]: rank + 1 for rank, idx in enumerate(sorted_indices)}
            
            # Calculate MDG for positive items
            for item_id, rating in item_ratings:
                if rating > 0:
                    rank = rankings[item_id]
                    if rank <= k:
                        mdg_scores[item_id].append(1.0 / np.log2(1 + rank))
                    else:
                        mdg_scores[item_id].append(0)
    
    # Calculate final MDG scores
    final_mdg = {}
    for item_id, gains in mdg_scores.items():
        n_users = len(item_users[item_id])
        if n_users > 0:
            final_mdg[item_id] = sum(gains) / n_users
    
    mdg_analysis = analyze_mdg_percentiles(final_mdg)
    return final_mdg, mdg_analysis

def mdg_calc_debiased(prior_model,
                     original_mf,
                     debiasing_model,
                     data_loader,
                     ml_data,
                     model_type=1,  # 0 for MF, 1 for DropoutNet
                     k=100,
                     total_eval_items=200,
                     device='cpu'):
    """Calculate MDG for debiased model using test items and sampled negatives"""
    user_ratings = defaultdict(list)
    item_users = defaultdict(list)
    mdg_scores = defaultdict(list)
    all_items = set(range(ml_data.n_items))
    
    # Group ratings
    for batch_triplet in data_loader:
        for i in range(len(batch_triplet[0])):
            user_id = batch_triplet[0][i].item()
            item_id = batch_triplet[1][i].item()
            rating = batch_triplet[2][i].item()
            user_ratings[user_id].append((item_id, rating))
            if rating > 0:
                item_users[item_id].append(user_id)
    
    prior_model.eval()
    debiasing_model.eval()
    if model_type == 1:
        original_mf.eval()
    
    with torch.no_grad():
        # Get base predictions with correct model combination
        if model_type == 0:  # Matrix Factorization
            u_emb, i_emb = prior_model.get_embeddings()
            u_emb = u_emb.to(device)
            i_emb = i_emb.to(device)
            R = torch.mm(i_emb, u_emb.t())
            
            all_users = torch.arange(ml_data.n_users, device=device)
            all_items_tensor = torch.arange(ml_data.n_items, device=device)
            user_biases = prior_model.user_bias(all_users).squeeze()
            item_biases = prior_model.item_bias(all_items_tensor).squeeze()
            R += user_biases.unsqueeze(0) + item_biases.unsqueeze(1) + prior_model.global_bias
            
        else:  # DropoutNet
            u_emb, i_emb = original_mf.get_embeddings()
            u_emb = u_emb.to(device)
            i_emb = i_emb.to(device)
            
            u_content = (torch.tensor(ml_data.user_content, dtype=torch.float32).to(device) 
                        if ml_data.user_content is not None else None)
            i_content = (torch.tensor(ml_data.item_content, dtype=torch.float32).to(device)
                        if ml_data.item_content is not None else None)
            
            u_encoded, i_encoded = prior_model.encode(
                u_emb,
                i_emb,
                u_content,
                i_content
            )
            R = torch.mm(i_encoded, u_encoded.t())
            
            all_users = torch.arange(ml_data.n_users, device=device)
            all_items_tensor = torch.arange(ml_data.n_items, device=device)
            user_biases = original_mf.user_bias(all_users).squeeze()
            item_biases = original_mf.item_bias(all_items_tensor).squeeze()
            R += user_biases.unsqueeze(0) + item_biases.unsqueeze(1) + original_mf.global_bias
        
        # Apply debiasing
        R = debiasing_model(R, is_training=False).preds
        
        for user_id, item_ratings in user_ratings.items():
            # Get test items and sample negatives
            test_items = set(item[0] for item in item_ratings)
            n_test = len(test_items)
            n_neg = max(0, total_eval_items - n_test)
            
            available_neg = list(all_items - test_items)
            if n_neg > 0 and available_neg:
                n_neg = min(n_neg, len(available_neg))
                neg_items = set(np.random.choice(available_neg, n_neg, replace=False))
            else:
                neg_items = set()
            
            eval_items = list(test_items | neg_items)
            
            # Get predictions and rankings
            preds = R[eval_items, user_id]
            sorted_indices = torch.argsort(preds, descending=True).cpu().numpy()
            rankings = {eval_items[idx]: rank + 1 for rank, idx in enumerate(sorted_indices)}
            
            # Calculate MDG for positive items
            for item_id, rating in item_ratings:
                if rating > 0:
                    rank = rankings[item_id]
                    if rank <= k:
                        mdg_scores[item_id].append(1.0 / np.log2(1 + rank))
                    else:
                        mdg_scores[item_id].append(0)
    
    # Calculate final MDG scores
    final_mdg = {}
    for item_id, gains in mdg_scores.items():
        n_users = len(item_users[item_id])
        if n_users > 0:
            final_mdg[item_id] = sum(gains) / n_users
    
    mdg_analysis = analyze_mdg_percentiles(final_mdg)
    return final_mdg, mdg_analysis
    
def ndcg_calc_base(model, data_loader, ml_data, k_values=[5, 10, 20, 50], device='cpu'):
    """Calculate NDCG using only test set items"""
    user_ratings = defaultdict(list)
   
    # Group ratings by user
    for batch_triplet in data_loader:
        for i in range(len(batch_triplet[0])):
            user_id, item_id, rating = batch_triplet[0][i], batch_triplet[1][i], batch_triplet[2][i]
            user_ratings[user_id.item()].append((item_id.item(), rating.item()))

    model.eval()
    ndcg_scores = []

    with torch.no_grad():
        for k_value in k_values:
            temp_ndcg = []
            
            for user_id, item_ratings in user_ratings.items():
                
                if len(item_ratings) < 2:
                    continue
                
                # Get only test items for this user
                test_items = torch.tensor([item[0] for item in item_ratings]).to(device)
                true_ratings = torch.tensor([item[1] for item in item_ratings])
                
                # Predict only test items
                user_tensor = torch.full((len(test_items),), user_id, dtype=torch.long).to(device)
                preds = model(user_tensor, test_items).preds.cpu()
                
                #Test binarize
                true_ratings = np.array([1 if num >= 4  else 0 for num in true_ratings])
                
                # Calculate NDCG on test items only
                ndcg = ndcg_score(
                    y_true=true_ratings.reshape(1, -1),
                    y_score=preds.reshape(1, -1),
                    k=min(k_value, len(test_items))
                )
                temp_ndcg.append(ndcg)
                
            ndcg_scores.append(np.mean(temp_ndcg))
    
    return ndcg_scores


def ndcg_calc_sampled(model, data_loader, ml_data, k_values=[5, 10, 20, 50], total_eval_items=200, device='cpu'):
   """
   Calculate NDCG using test items plus sampled negative items
   total_eval_items: Total number of items to evaluate per user (test items + sampled negatives)
   """
   user_ratings = defaultdict(list)
   all_items = set(range(ml_data.n_items))
   
   # Collect test items per user
   for batch_triplet in data_loader:
       for i in range(len(batch_triplet[0])):
           user_id, item_id, rating = batch_triplet[0][i], batch_triplet[1][i], batch_triplet[2][i]
           user_ratings[user_id.item()].append((item_id.item(), rating.item()))
           
           
   
   model.eval()
   ndcg_scores = []
   precision_scores = []
   recall_scores = []
   
   with torch.no_grad():
       for k_value in k_values:
           temp_ndcg = []
           temp_precision = []
           temp_recall = []
           
           for user_id, item_ratings in user_ratings.items():
               if len(item_ratings) < 2:
                   continue
               
               # Get positive (test) items
               test_items = set(item[0] for item in item_ratings)
               
               # Calculate how many negative samples needed
               n_test = len(test_items)
               n_neg = max(0, total_eval_items - n_test)
               
               # Sample negative items
               available_neg = list(all_items - test_items)
               if n_neg > 0 and len(available_neg) > 0:
                   n_neg = min(n_neg, len(available_neg))
                   neg_items = set(np.random.choice(available_neg, n_neg, replace=False))
               else:
                   neg_items = set()
               
               # Combine positive and negative items
               eval_items = list(test_items | neg_items)
               eval_items_tensor = torch.tensor(eval_items).to(device)
               
               # Create true ratings (1 for relevant test items, 0 for others)
               true_ratings = np.zeros(len(eval_items))
               for idx, item in enumerate(eval_items):
                   if item in test_items:
                       rating = [r[1] for r in item_ratings if r[0] == item][0]
                       true_ratings[idx] = 1 if rating >= 4 else 0
               
               # Get predictions
               user_tensor = torch.full((len(eval_items),), user_id, dtype=torch.long).to(device)
               preds = model(user_tensor, eval_items_tensor).preds.cpu().numpy()
               
               # Calculate metrics
               if len(eval_items) >= 2:  # Need at least 2 items for NDCG
                   ndcg = ndcg_score(
                       y_true=true_ratings.reshape(1, -1),
                       y_score=preds.reshape(1, -1),
                       k=min(k_value, len(eval_items))
                   )
                   temp_ndcg.append(ndcg)
               
                   # Calculate Precision and Recall
                   total_relevant = np.sum(true_ratings)
                   if total_relevant > 0:
                       topk = min(k_value, len(eval_items))
                       top_indices = np.argsort(-preds)[:topk]
                       relevant_at_k = np.sum(true_ratings[top_indices])
                       
                       precision = relevant_at_k / topk
                       recall = relevant_at_k / total_relevant
                       
                       temp_precision.append(precision)
                       temp_recall.append(recall)
           
           if temp_ndcg:
               ndcg_scores.append(np.mean(temp_ndcg))
               precision_scores.append(np.mean(temp_precision))
               recall_scores.append(np.mean(temp_recall))
           else:
               ndcg_scores.append(0)
               precision_scores.append(0)
               recall_scores.append(0)
   
   return ndcg_scores, precision_scores, recall_scores

def ndcg_calc_dropout(model,
                     dropout_model,
                     test_loader,
                     ml_data,
                     ks=[15,30],
                     device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Calculate NDCG@k for dropout model with proper embedding and bias handling
    """
    dropout_model.eval()
    model.eval()
    
    # Get all users in test set
    user_ratings = defaultdict(list)
    item_users = defaultdict(list)
    for batch_triplet in test_loader:
        for i in range(len(batch_triplet[0])):
            user_id, item_id, rating = batch_triplet[0][i], batch_triplet[1][i], batch_triplet[2][i]
            user_ratings[user_id.item()].append((item_id.item(), rating.item()))
            if rating.item() > 0:
                item_users[item_id.item()].append(user_id.item())
    
    ndcg_scores = []

    
    # Get base embeddings once
    with torch.no_grad():
        u_emb, i_emb = model.get_embeddings()
        u_emb = u_emb.to(device)
        i_emb = i_emb.to(device)
    
    # Prepare content features once if used
    u_content = None
    i_content = None
    if ml_data.user_content is not None:
        u_content = torch.tensor(ml_data.user_content, dtype=torch.float32).to(device)
    if ml_data.item_content is not None:
        i_content = torch.tensor(ml_data.item_content, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        for k_value in ks:
            temp_ndcg = []
            
            for user_id, items_ratings in tqdm(user_ratings.items()):
                # Create ground truth ratings vector
                true_ratings = np.zeros(ml_data.n_items)
                for item_id, rating in items_ratings:
                    true_ratings[item_id] = rating
                
                # Get user embedding from base embeddings
                user_emb = u_emb[user_id:user_id+1]
                
                # Get encoded embeddings using actual base embeddings
                u_encoded, i_encoded = dropout_model.encode(
                    user_emb,
                    i_emb,
                    u_content[user_id:user_id+1] if u_content is not None else None,
                    i_content
                )
                
                # Calculate dot product with transformed embeddings
                dot_products = torch.mm(i_encoded, u_encoded.t()).squeeze()
                
                # Add bias terms from base model
                user_bias = model.user_bias(torch.tensor([user_id], device=device)).squeeze()
                item_biases = model.item_bias(torch.arange(ml_data.n_items, device=device)).squeeze()
                pred_ratings = dot_products + user_bias + item_biases + model.global_bias
                # Calculate NDCG
                ndcg = ndcg_score(
                    y_true=true_ratings.reshape(1, -1),
                    y_score=pred_ratings.cpu().numpy().reshape(1, -1),
                    k=k_value
                )
                
                if not np.isnan(ndcg):
                    temp_ndcg.append(ndcg)
                
            ndcg_scores.append(np.mean(temp_ndcg))
            
    return ndcg_scores

def ndcg_calc_debiased(prior_model,
                      mf_model,
                      debiasing_model, 
                      data_loader, 
                      ml_data,
                      model_type=1,  # 0 for MF, 1 for DropoutNet
                      k_values=[15, 30], 
                      device='cpu'):
    """
    Calculate NDCG using debiased rankings with proper model handling
    """
    # Get all users in test set
    user_ratings = defaultdict(list)
    item_users = defaultdict(list)
    for batch_triplet in data_loader:
        for i in range(len(batch_triplet[0])):
            user_id, item_id, rating = batch_triplet[0][i], batch_triplet[1][i], batch_triplet[2][i]
            user_ratings[user_id.item()].append((item_id.item(), rating.item()))
            if rating.item() > 0:
                item_users[item_id.item()].append(user_id.item())
    
    prior_model.eval()
    debiasing_model.eval()
    if model_type == 1:
        mf_model.eval()
        
    ndcg_scores = []
    mdg_scores = defaultdict(list)
    precision_scores = []
    recall_scores = []
    
    # Get base predictions with correct model combination
    with torch.no_grad():
        if model_type == 0:  # Matrix Factorization
            u_emb, i_emb = prior_model.get_embeddings()
            u_emb = u_emb.to(device)
            i_emb = i_emb.to(device)
            R_base = torch.mm(i_emb, u_emb.t())
            
            # Get bias terms from same model
            all_users = torch.arange(ml_data.n_users, device=device)
            all_items = torch.arange(ml_data.n_items, device=device)
            user_biases = prior_model.user_bias(all_users).squeeze()
            item_biases = prior_model.item_bias(all_items).squeeze()
            global_bias = prior_model.global_bias
            
        else:  # DropoutNet
            # Get base embeddings from original MF for DropoutNet input
            u_emb, i_emb = mf_model.get_embeddings()
            u_emb = u_emb.to(device)
            i_emb = i_emb.to(device)
            
            # Get content features if needed
            u_content = (torch.tensor(ml_data.user_content, dtype=torch.float32).to(device) 
                        if ml_data.user_content is not None else None)
            i_content = (torch.tensor(ml_data.item_content, dtype=torch.float32).to(device)
                        if ml_data.item_content is not None else None)
            
            # Get transformed embeddings from DropoutNet
            u_encoded, i_encoded = prior_model.encode(
                u_emb,
                i_emb,
                u_content,
                i_content
            )
            R_base = torch.mm(i_encoded, u_encoded.t())
            
            # Get bias terms from original MF model
            all_users = torch.arange(ml_data.n_users, device=device)
            all_items = torch.arange(ml_data.n_items, device=device)
            user_biases = mf_model.user_bias(all_users).squeeze()
            item_biases = mf_model.item_bias(all_items).squeeze()
            global_bias = mf_model.global_bias
        
        # Add bias terms to base predictions
        R = R_base + user_biases.unsqueeze(0) + item_biases.unsqueeze(1) + global_bias
        
        # Apply debiasing for fairness
        debiased_R = debiasing_model(R, is_training=False).preds
        
        for k_value in k_values:
            temp_ndcg = []
            temp_precision = []
            temp_recall = []
            
            for user_id, item_ratings in tqdm(user_ratings.items()):
                # Create ground truth ratings vector
                true_ratings = np.zeros(ml_data.n_items)
                for item_id, rating in item_ratings:
                    true_ratings[item_id] = rating
                
                # Get debiased predictions for this user
                preds = debiased_R[:, user_id]
                
                # Binarize true ratings (if needed for your use case)
                true_ratings = np.array([1 if num >= 4 else 0 for num in true_ratings])
                
                # Calculate Precision and Recall
                total_relevant_items = np.sum(true_ratings)
                if total_relevant_items == 0:
                    continue
                    
                topk_res = torch.topk(preds, k=k_value)[1]
                relevant_count = np.sum(true_ratings[topk_res.cpu()])
                
                temp_precision.append(relevant_count / k_value)
                temp_recall.append(relevant_count / total_relevant_items)
                
                # Calculate NDCG
                ndcg = ndcg_score(
                    y_true=true_ratings.reshape(1, -1),
                    y_score=preds.cpu().numpy().reshape(1, -1),
                    k=k_value
                )
                
                if not np.isnan(ndcg):
                    temp_ndcg.append(ndcg)
                    
                
            
            ndcg_scores.append(np.mean(temp_ndcg))
            precision_scores.append(np.mean(temp_precision))
            recall_scores.append(np.mean(temp_recall))
    
    
    return ndcg_scores, precision_scores, recall_scores

def calculate_mdg(rankings, positive_items, k=100):
    """Calculate Mean Discount Gain for positive items up to rank k"""
    gains = []
    for item_id in positive_items:
        rank = rankings.get(item_id)
        if rank is not None and rank <= k:
            gains.append(1.0 / np.log2(1 + rank))
    return np.mean(gains) if gains else 0


def ndcg_calc_dropout_sampled(base_model, 
                             model,
                             data_loader,
                             ml_data,
                             k_values=[5, 10, 20, 50],
                             total_eval_items=200,
                             device='cpu'):
    """Calculate NDCG using test items plus sampled negatives for dropout model"""
    model.eval()
    base_model.eval()
    
    user_ratings = defaultdict(list)
    item_users = defaultdict(list)
    all_items = set(range(ml_data.n_items))
    
    for batch_triplet in data_loader:
        for i in range(len(batch_triplet[0])):
            user_id, item_id, rating = batch_triplet[0][i], batch_triplet[1][i], batch_triplet[2][i]
            user_ratings[user_id.item()].append((item_id.item(), rating.item()))
            if rating.item() > 0:
                item_users[item_id.item()].append(user_id.item())
    
    ndcg_scores = []
    precision_scores = []
    recall_scores = []
    
    # Get base embeddings once
    with torch.no_grad():
        u_emb, i_emb = base_model.get_embeddings()
        u_emb = u_emb.to(device)
        i_emb = i_emb.to(device)
    
    # Prepare content features
    u_content = None
    i_content = None
    if ml_data.user_content is not None:
        u_content = torch.tensor(ml_data.user_content, dtype=torch.float32).to(device)
    if ml_data.item_content is not None:
        i_content = torch.tensor(ml_data.item_content, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        for k_value in k_values:
            temp_ndcg = []
            temp_precision = []
            temp_recall = []
            
            for user_id, item_ratings in user_ratings.items():
                if len(item_ratings) < 2:
                    continue
                
                # Get test items and sample negatives
                test_items = set(item[0] for item in item_ratings)
                n_test = len(test_items)
                n_neg = max(0, total_eval_items - n_test)
                
                available_neg = list(all_items - test_items)
                if n_neg > 0 and available_neg:
                    n_neg = min(n_neg, len(available_neg))
                    neg_items = set(np.random.choice(available_neg, n_neg, replace=False))
                else:
                    neg_items = set()
                
                eval_items = list(test_items | neg_items)
                
                # Get user embedding and encode
                user_emb = u_emb[user_id:user_id+1]
                items_emb = i_emb[eval_items]
                
                u_encoded, i_encoded = model.encode(
                    user_emb,
                    items_emb,
                    u_content[user_id:user_id+1] if u_content is not None else None,
                    i_content[eval_items] if i_content is not None else None
                )
                
                # Get predictions through matrix multiplication
                preds = torch.mm(i_encoded, u_encoded.t()).squeeze()
                
                # Add bias terms
                user_bias = base_model.user_bias(torch.tensor([user_id], device=device)).squeeze()
                item_biases = base_model.item_bias(torch.tensor(eval_items, device=device)).squeeze()
                preds = preds + user_bias + item_biases + base_model.global_bias
                
                # Create true ratings vector
                true_ratings = np.zeros(len(eval_items))
                for idx, item in enumerate(eval_items):
                    if item in test_items:
                        rating = [r[1] for r in item_ratings if r[0] == item][0]
                        true_ratings[idx] = 1 if rating >= 4 else 0
                
                if len(eval_items) >= 2:
                    ndcg = ndcg_score(
                        y_true=true_ratings.reshape(1, -1),
                        y_score=preds.cpu().numpy().reshape(1, -1),
                        k=min(k_value, len(eval_items))
                    )
                    temp_ndcg.append(ndcg)
                    
                    # Calculate Precision and Recall
                    total_relevant = np.sum(true_ratings)
                    if total_relevant > 0:
                        topk = min(k_value, len(eval_items))
                        top_indices = torch.topk(preds, k=topk)[1].cpu().numpy()
                        relevant_at_k = np.sum(true_ratings[top_indices])
                        
                        precision = relevant_at_k / topk
                        recall = relevant_at_k / total_relevant
                        
                        temp_precision.append(precision)
                        temp_recall.append(recall)
            
            if temp_ndcg:
                ndcg_scores.append(np.mean(temp_ndcg))
                precision_scores.append(np.mean(temp_precision))
                recall_scores.append(np.mean(temp_recall))
            else:
                ndcg_scores.append(0)
                precision_scores.append(0)
                recall_scores.append(0)
    
    return ndcg_scores, precision_scores, recall_scores


def ndcg_calc_debiased_sampled(prior_model,
                              original_mf,
                              model,
                              data_loader,
                              ml_data,
                              model_type=1,
                              k_values=[5, 10, 20, 50],
                              total_eval_items=200,
                              device='cpu'):
    """Calculate NDCG using test items plus sampled negatives for debiased model"""
    prior_model.eval()
    model.eval()
    if model_type == 1:
        original_mf.eval()
    
    user_ratings = defaultdict(list)
    all_items = set(range(ml_data.n_items))
    item_users = defaultdict(list)
    
    for batch_triplet in data_loader:
        for i in range(len(batch_triplet[0])):
            user_id, item_id, rating = batch_triplet[0][i], batch_triplet[1][i], batch_triplet[2][i]
            user_ratings[user_id.item()].append((item_id.item(), rating.item()))
            if rating.item() > 0:
                item_users[item_id.item()].append(user_id.item())
    
    ndcg_scores = []
    precision_scores = []
    recall_scores = []
    
    with torch.no_grad():
        # Get base predictions with correct model combination
        if model_type == 0:  # Matrix Factorization
            u_emb, i_emb = prior_model.get_embeddings()
            u_emb = u_emb.to(device)
            i_emb = i_emb.to(device)
            R = torch.mm(i_emb, u_emb.t())
            
            # Add bias terms
            all_users = torch.arange(ml_data.n_users, device=device)
            all_items_tensor = torch.arange(ml_data.n_items, device=device)
            user_biases = prior_model.user_bias(all_users).squeeze()
            item_biases = prior_model.item_bias(all_items_tensor).squeeze()
            R += user_biases.unsqueeze(0) + item_biases.unsqueeze(1) + prior_model.global_bias
            
        else:  # DropoutNet
            u_emb, i_emb = original_mf.get_embeddings()
            u_emb = u_emb.to(device)
            i_emb = i_emb.to(device)
            
            u_content = (torch.tensor(ml_data.user_content, dtype=torch.float32).to(device) 
                        if ml_data.user_content is not None else None)
            i_content = (torch.tensor(ml_data.item_content, dtype=torch.float32).to(device)
                        if ml_data.item_content is not None else None)
            
            u_encoded, i_encoded = prior_model.encode(
                u_emb,
                i_emb,
                u_content,
                i_content
            )
            R = torch.mm(i_encoded, u_encoded.t())
            
            # Add bias terms from original MF
            all_users = torch.arange(ml_data.n_users, device=device)
            all_items_tensor = torch.arange(ml_data.n_items, device=device)
            user_biases = original_mf.user_bias(all_users).squeeze()
            item_biases = original_mf.item_bias(all_items_tensor).squeeze()
            R += user_biases.unsqueeze(0) + item_biases.unsqueeze(1) + original_mf.global_bias
        
        # Apply debiasing
        R = model(R, is_training=False).preds
        
        for k_value in k_values:
            temp_ndcg = []
            temp_precision = []
            temp_recall = []
            
            for user_id, item_ratings in user_ratings.items():
                if len(item_ratings) < 2:
                    continue
                
                # Get test items and sample negatives
                test_items = set(item[0] for item in item_ratings)
                n_test = len(test_items)
                n_neg = max(0, total_eval_items - n_test)
                
                available_neg = list(all_items - test_items)  # Now both are sets
                if n_neg > 0 and available_neg:
                    n_neg = min(n_neg, len(available_neg))
                    neg_items = set(np.random.choice(available_neg, n_neg, replace=False))
                else:
                    neg_items = set()
                
                eval_items = list(test_items | neg_items)
                
                # Get predictions for evaluation items
                preds = R[eval_items, user_id]
                
                # Create true ratings vector
                true_ratings = np.zeros(len(eval_items))
                for idx, item in enumerate(eval_items):
                    if item in test_items:
                        rating = [r[1] for r in item_ratings if r[0] == item][0]
                        true_ratings[idx] = 1 if rating >= 4 else 0
                
                if len(eval_items) >= 2:
                    ndcg = ndcg_score(
                        y_true=true_ratings.reshape(1, -1),
                        y_score=preds.cpu().numpy().reshape(1, -1),
                        k=min(k_value, len(eval_items))
                    )
                    temp_ndcg.append(ndcg)
                    
                    # Calculate Precision and Recall
                    total_relevant = np.sum(true_ratings)
                    if total_relevant > 0:
                        topk = min(k_value, len(eval_items))
                        top_indices = torch.topk(preds, k=topk)[1].cpu().numpy()
                        relevant_at_k = np.sum(true_ratings[top_indices])
                        
                        precision = relevant_at_k / topk
                        recall = relevant_at_k / total_relevant
                        
                        temp_precision.append(precision)
                        temp_recall.append(recall)
            
            if temp_ndcg:
                ndcg_scores.append(np.mean(temp_ndcg))
                precision_scores.append(np.mean(temp_precision))
                recall_scores.append(np.mean(temp_recall))
            else:
                ndcg_scores.append(0)
                precision_scores.append(0)
                recall_scores.append(0)
    
    return ndcg_scores, precision_scores, recall_scores
def create_stats(scores_dict):
        if not scores_dict:
            return {
                'n_items': 0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
            }
        
        scores = np.array(list(scores_dict.values()))
        items = np.array(list(scores_dict.keys()))
        
        stats = {
            'n_items': len(scores),
            'mean': float(np.mean(scores)),
            'median': float(np.median(scores)),
            'std': float(np.std(scores)),
            'percentiles': {}
        }
        
        return stats
def get_item_split(train_data,
                  test_data,
                  min_interactions=5,
                  verbose=True):
    """
    Split items into cold and warm based on training data presence
    
    Args:
        train_data: Training DataFrame
        test_data: Test DataFrame
        min_interactions: Minimum number of interactions to consider an item warm
        verbose: Whether to print statistics
        
    Returns:
        Tuple of (cold_items, warm_items) sets
    """
    # Count item interactions in training data
    train_item_counts = train_data['item_idx'].value_counts()
    
    # Get items with sufficient interactions
    warm_items = set(train_item_counts[train_item_counts >= min_interactions].index)
    
    # Get all test items
    test_items = set(test_data['item_idx'].unique())
    
    # Split into cold and warm
    cold_items = test_items - warm_items
    warm_items = test_items & warm_items
    
    if verbose:
        total_items = len(test_items)
        print(f"\nItem Split Analysis:")
        print(f"Total items in test set: {total_items}")
        print(f"Found {len(cold_items)} cold items and {len(warm_items)} warm items")
        print(f"Cold item ratio: {len(cold_items)/total_items:.2%}")
        
        # Additional statistics
        train_items = set(train_data['item_idx'].unique())
        print("\nDetailed Statistics:")
        print(f"Total unique items in training: {len(train_items)}")
        print(f"Total unique items in test: {len(test_items)}")
        print(f"Items in test but not in training: {len(test_items - train_items)}")
        print(f"Items with insufficient interactions: {len(test_items - warm_items - (test_items - train_items))}")
        
        # Interaction statistics
        print("\nInteraction Statistics:")
        test_item_counts = test_data['item_idx'].value_counts()
        print(f"Average interactions per cold item: {test_item_counts[list(cold_items)].mean():.2f}")
        print(f"Average interactions per warm item: {test_item_counts[list(warm_items)].mean():.2f}")
        
        # Distribution of interaction counts
        cold_interactions = train_item_counts[list(cold_items & train_items)].value_counts().sort_index()
        if not cold_interactions.empty:
            print("\nCold items training interaction distribution:")
            for count, num_items in cold_interactions.items():
                print(f"  {count} interactions: {num_items} items")
    
    return cold_items, warm_items

def analyze_mdg_with_splits(mdg_scores,
                          cold_items,
                          warm_items):
    """
    Analyze MDG scores using predefined cold/warm item splits
    
    Args:
        mdg_scores: Dictionary of item MDG scores
        cold_items: Set of cold item indices
        warm_items: Set of warm item indices
        
    Returns:
        Dictionary containing analysis results
    """
    # Split scores by cold/warm
    cold_mdg = {item: score for item, score in mdg_scores.items() if item in cold_items}
    warm_mdg = {item: score for item, score in mdg_scores.items() if item in warm_items}
    
    results = {
        'overall': create_stats(mdg_scores),
        'warm': create_stats(warm_mdg),
        'cold': create_stats(cold_mdg),
        'coverage': {
            'total_items_with_mdg': len(mdg_scores),
            'cold_items_with_mdg': len(cold_mdg),
            'warm_items_with_mdg': len(warm_mdg),
            'cold_items_missing_mdg': len(cold_items - set(mdg_scores.keys())),
            'warm_items_missing_mdg': len(warm_items - set(mdg_scores.keys()))
        }
    }
    
    # Add comparison if both cold and warm items exist
    if cold_mdg and warm_mdg:
        cold_scores = np.array(list(cold_mdg.values()))
        warm_scores = np.array(list(warm_mdg.values()))
        
        results['comparison'] = {
            'cold_vs_warm_mean_diff': float(np.mean(cold_scores) - np.mean(warm_scores)),
            'cold_vs_warm_median_diff': float(np.median(cold_scores) - np.median(warm_scores)),
        }
    
    return results
        
def print_mdg_analysis(analysis):
    """
    Print MDG analysis results including coverage statistics
    
    Args:
        analysis: Output from analyze_mdg_with_splits
    """
    print("\nMDG Analysis Summary:")
    print("-" * 50)
    
    # Print coverage statistics
    cov = analysis['coverage']
    print("\nCoverage Statistics:")
    print(f"Total items with MDG scores: {cov['total_items_with_mdg']}")
    print(f"Cold items with scores: {cov['cold_items_with_mdg']}")
    print(f"Warm items with scores: {cov['warm_items_with_mdg']}")
    print(f"Cold items missing scores: {cov['cold_items_missing_mdg']}")
    print(f"Warm items missing scores: {cov['warm_items_missing_mdg']}")
    
    # Print basic statistics for each category
    for category in ['overall', 'warm', 'cold']:
        stats = analysis[category]
        if stats['n_items'] > 0:
            print(f"\n{category.upper()} Items:")
            print(f"Number of items: {stats['n_items']}")
            print(f"Mean MDG: {stats['mean']:.4f}")
            print(f"Median MDG: {stats['median']:.4f}")
            print(f"Std Dev: {stats['std']:.4f}")
    
    # Print comparison statistics if available
    if 'comparison' in analysis:
        comp = analysis['comparison']
        print("\nComparison Statistics:")
        print(f"Cold vs Warm Mean Difference: {comp['cold_vs_warm_mean_diff']:.4f}")
        print(f"Cold vs Warm Median Difference: {comp['cold_vs_warm_median_diff']:.4f}")
        

@dataclass
class EvalMetrics:
    """Container for evaluation metrics"""
    ndcg: List[float]
    precision: List[float]
    recall: List[float]
    
@dataclass
class SplitMetrics:
    """Container for cold/warm split metrics"""
    cold_users: EvalMetrics
    warm_users: EvalMetrics
    all_users: EvalMetrics
    n_cold_users: int
    n_warm_users: int

def evaluate_split(eval_model,
                  test_loader,
                  ml_data,
                  cold_users,
                  warm_users,
                  k_values,
                  evaluation_func,
                  **kwargs):
    """
    Evaluate model separately on cold and warm users
    
    Args:
        model: Base model for embeddings
        test_loader: Test data loader
        ml_data: MovieLens data container
        cold_users: Set of cold user indices
        warm_users: Set of warm user indices
        k_values: List of k values for NDCG
        evaluation_func: Evaluation function to use
        **kwargs: Additional arguments for evaluation function
    """
    # Create separate DataFrames for cold and warm users
    test_data = ml_data.test_data.copy()
    
    cold_test_data = test_data[test_data['user_idx'].isin(cold_users)]
    warm_test_data = test_data[test_data['user_idx'].isin(warm_users)]
    
    # Create new datasets and loaders
    cold_dataset = MovieLensDataset(cold_test_data)
    warm_dataset = MovieLensDataset(warm_test_data)
    
    cold_loader = DataLoader(
        cold_dataset,
        batch_size=test_loader.batch_size,
        shuffle=False,
        num_workers=test_loader.num_workers,
        pin_memory=test_loader.pin_memory
    )
    
    warm_loader = DataLoader(
        warm_dataset,
        batch_size=test_loader.batch_size,
        shuffle=False,
        num_workers=test_loader.num_workers,
        pin_memory=test_loader.pin_memory
    )
    
    # Handle empty splits
    if len(cold_test_data) > 0:
        print("Evaluating cold users...")
        cold_ndcgs, cold_prec, cold_rec = evaluation_func(
            model=eval_model,  
            data_loader=cold_loader,
            ml_data=ml_data,
            k_values=k_values,
            **kwargs
        )
    else:
        print("No cold users found, skipping evaluation")
        cold_ndcgs = [0.0] * len(k_values)
        cold_prec = [0.0] * len(k_values)
        cold_rec = [0.0] * len(k_values)
    
    if len(warm_test_data) > 0:
        print("Evaluating warm users...")
        warm_ndcgs, warm_prec, warm_rec = evaluation_func(
            model=eval_model,  
            data_loader=warm_loader,
            ml_data=ml_data,
            k_values=k_values,
            **kwargs
        )
    else:
        print("No warm users found, skipping evaluation")
        warm_ndcgs = [0.0] * len(k_values)
        warm_prec = [0.0] * len(k_values)
        warm_rec = [0.0] * len(k_values)
    
    print("Evaluating all users...")
    all_ndcgs, all_prec, all_rec = evaluation_func(
        model=eval_model,  
        data_loader=test_loader,
        ml_data=ml_data,
        k_values=k_values,
        **kwargs
    )
    
    return SplitMetrics(
        cold_users=EvalMetrics(cold_ndcgs, cold_prec, cold_rec),
        warm_users=EvalMetrics(warm_ndcgs, warm_prec, warm_rec),
        all_users=EvalMetrics(all_ndcgs, all_prec, all_rec),
        n_cold_users=len(cold_test_data['user_idx'].unique()),
        n_warm_users=len(warm_test_data['user_idx'].unique())
    )


def get_user_split(train_data, 
                   test_data,
                   min_interactions = 5):
    """
    Split users into cold and warm based on training data presence
    
    Args:
        train_data: Training DataFrame
        test_data: Test DataFrame
        min_interactions: Minimum number of interactions to consider a user warm
        
    Returns:
        Tuple of (cold_users, warm_users) sets
    """
    # Count user interactions in training data
    train_user_counts = train_data['user_idx'].value_counts()
    
    # Get users with sufficient interactions
    warm_users = set(train_user_counts[train_user_counts >= min_interactions].index)
    
    # Get all test users
    test_users = set(test_data['user_idx'].unique())
    
    # Split into cold and warm
    cold_users = test_users - warm_users
    warm_users = test_users & warm_users
    
    print(f"Found {len(cold_users)} cold users and {len(warm_users)} warm users in test set")
    print(f"Cold user ratio: {len(cold_users) / len(test_users):.2%}")
    
    return cold_users, warm_users