import numpy as np
import torch
from sklearn.metrics import ndcg_score
import scipy
from collections import defaultdict
from tqdm import tqdm

def analyze_mdg_percentiles(mdg_scores, percentiles = [10, 20, 90]):
    """
    Analyze MDG scores for specific percentiles of items
    
    Args:
        mdg_scores: Dictionary mapping item IDs to MDG scores
        percentiles: List of percentiles to analyze (e.g., [10, 20, 90] for bottom 10%, 20%, top 10%)
    
    Returns:
        Dictionary containing average MDG for each percentile group
    """
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


# def ndcg_calc_base(model, data_loader, ml_data, k_values=[5, 10, 20, 50], device='cpu'):
#     """
#     Calculate NDCG at different K values for base MF model
#     Args:
#         model: BiasedMF model
#         data_loader: Test data loader
#         ml_data: MovieLens data container
#         k_values: List of K values for NDCG calculation
#         device: Computing device
#     Returns:
#         Array of NDCG scores for each K
#     """
#     # Get all users in test set
#     user_ratings = defaultdict(list)
#     item_users = defaultdict(list)
#     for batch_triplet in data_loader:
#         for i in range(len(batch_triplet[0])):
#             user_id, item_id, rating = batch_triplet[0][i], batch_triplet[1][i], batch_triplet[2][i]
#             user_ratings[user_id.item()].append((item_id.item(), rating.item()))
#             if rating.item() > 0:
#                 item_users[item_id.item()].append(user_id.item())
    
#     model.eval()
#     ndcg_scores = []
#     mdg_scores = defaultdict(list)
#     precision_scores = []
#     recall_scores = []
    
    
#     all_items = torch.arange(ml_data.n_items, device=device)
    
#     with torch.no_grad():
#         for k_value in k_values:
#             temp_ndcg = []
#             temp_precision = []
#             temp_recall = []
#             for user_id, item_ratings in tqdm(user_ratings.items(), 
#                                           desc=f'Processing NDCG@{k_value}'):
#                 # Create ground truth ratings vector (zeros for non-test items)
#                 true_ratings = np.zeros(ml_data.n_items)
#                 for item_id, rating in item_ratings:
#                     true_ratings[item_id] = rating
                
#                 # Get predictions for ALL items
#                 user_tensor = torch.full((ml_data.n_items,), user_id, dtype=torch.long).to(device)
#                 preds = model(user_tensor, all_items).preds.cpu().numpy()
#                 print(f"TESTING {preds.shape}")
                
                
#                 #Test binarize
#                 true_ratings = np.array([1 if num >= 4  else 0 for num in true_ratings])
                
#                 #Calculate Precision and Recall
#                 total_relevant_items = np.sum(true_ratings)
#                 if(total_relevant_items == 0):
#                     continue
                
#                 topk_res = torch.topk(torch.from_numpy(preds), k=k_value)[1]
#                 relevant_count = np.sum(true_ratings[topk_res])
                
#                 temp_precision.append(relevant_count / k_value)
#                 temp_recall.append(relevant_count / total_relevant_items)
                
                
                
#                 # Calculate NDCG
#                 ndcg = ndcg_score(
#                     y_true=true_ratings.reshape(1, -1),
#                     y_score=preds.reshape(1, -1),
#                     k=k_value
#                 )                   
#                 temp_ndcg.append(ndcg)
                
#                 # Calculate rankings for MDG
#                 sorted_indices = np.argsort(-preds)
#                 rankings = {item_id: rank + 1 for rank, item_id in enumerate(sorted_indices)}
                
#                 # Update MDG scores
#                 for item_id, rating in item_ratings:
#                     if rating > 0:
#                         rank = rankings[item_id]
#                         if rank <= 100:
#                             mdg_scores[item_id].append(1.0 / np.log2(1 + rank))
                            
            
#             ndcg_scores.append(np.mean(temp_ndcg))
#             precision_scores.append(np.mean(temp_precision))
#             recall_scores.append(np.mean(temp_recall))
#         # Calculate final MDG scores
#     final_mdg = {}
#     for item_id, gains in mdg_scores.items():
#         matched_users = len(item_users[item_id])
#         if matched_users > 0:
#             final_mdg[item_id] = sum(gains) / matched_users
    
#     # Analyze MDG percentiles
#     mdg_analysis = analyze_mdg_percentiles(final_mdg)
    
#     return ndcg_scores, precision_scores, recall_scores, final_mdg, mdg_analysis
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
   
   return ndcg_scores, 0, 0, 0, 0

def ndcg_calc_sampled(model, data_loader, ml_data, k_values=[5, 10, 20, 50], total_eval_items=1000, device='cpu'):
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
   
   return ndcg_scores, precision_scores, recall_scores, 0, 0

def get_rankings(predictions):
    """
    Get complete rankings for all items from predictions
    
    Args:
        predictions: Array of prediction scores
        
    Returns:
        Dictionary mapping item indices to their ranks (1-based)
    """
    sorted_indices = np.argsort(-predictions)  # Descending order
    for rank, item_id in enumerate(sorted_indices):
        print(item_id)
    return {item_id: rank + 1 for rank, item_id in enumerate(sorted_indices)}

def ndcg_calc_dropout(base_model,
                     model,
                     test_loader,
                     ml_data,
                     ks=[15,30],
                     device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Calculate NDCG@k for dropout model with proper embedding and bias handling
    """
    model.eval()
    base_model.eval()
    
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
        u_emb, i_emb = base_model.get_embeddings()
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
                u_encoded, i_encoded = model.encode(
                    user_emb,
                    i_emb,
                    u_content[user_id:user_id+1] if u_content is not None else None,
                    i_content
                )
                
                # Calculate dot product with transformed embeddings
                dot_products = torch.mm(i_encoded, u_encoded.t()).squeeze()
                
                # Add bias terms from base model
                user_bias = base_model.user_bias(torch.tensor([user_id], device=device)).squeeze()
                item_biases = base_model.item_bias(torch.arange(ml_data.n_items, device=device)).squeeze()
                pred_ratings = dot_products + user_bias + item_biases + base_model.global_bias
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
                      original_mf,
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
        original_mf.eval()
        
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
            u_emb, i_emb = original_mf.get_embeddings()
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
            user_biases = original_mf.user_bias(all_users).squeeze()
            item_biases = original_mf.item_bias(all_items).squeeze()
            global_bias = original_mf.global_bias
        
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
                    
                # Calculate rankings for MDG
                # sorted_indices = np.argsort(-preds)
                # rankings = {item_id: rank + 1 for rank, item_id in enumerate(sorted_indices)}
                
                rankings = get_rankings(preds)
                
            
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
                             total_eval_items=1000,
                             device='cpu'):
    """Calculate NDCG using test items plus sampled negatives for dropout model"""
    model.eval()
    base_model.eval()
    
    user_ratings = defaultdict(list)
    all_items = set(range(ml_data.n_items))
    
    for batch_triplet in data_loader:
        for i in range(len(batch_triplet[0])):
            user_id, item_id, rating = batch_triplet[0][i], batch_triplet[1][i], batch_triplet[2][i]
            user_ratings[user_id.item()].append((item_id.item(), rating.item()))
    
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
                              debiasing_model,
                              data_loader,
                              ml_data,
                              model_type=1,
                              k_values=[5, 10, 20, 50],
                              total_eval_items=1000,
                              device='cpu'):
    """Calculate NDCG using test items plus sampled negatives for debiased model"""
    prior_model.eval()
    debiasing_model.eval()
    if model_type == 1:
        original_mf.eval()
    
    user_ratings = defaultdict(list)
    all_items = set(range(ml_data.n_items))  # Changed to set of integers
    
    for batch_triplet in data_loader:
        for i in range(len(batch_triplet[0])):
            user_id, item_id, rating = batch_triplet[0][i], batch_triplet[1][i], batch_triplet[2][i]
            user_ratings[user_id.item()].append((item_id.item(), rating.item()))
    
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
        R = debiasing_model(R, is_training=False).preds
        
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
    
    return ndcg_scores, precision_scores, recall_scores, 0, 0
