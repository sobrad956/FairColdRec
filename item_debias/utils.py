import time
import datetime
import numpy as np
from scipy import stats
import torch
from torch.utils.data import Dataset
import copy
from typing import List, Tuple, Dict, Optional
import scipy.sparse


class Timer:
    """Timer object to record running time of functions"""
    def __init__(self, name: str = 'default'):
        self._start_time = None
        self._name = name
        self.tic()

    def tic(self):
        """Start the timer"""
        self._start_time = time.time()
        return self

    def toc(self, message: Optional[str] = None) -> 'Timer':
        """Stop the timer and print elapsed time"""
        elapsed = time.time() - self._start_time
        message = '' if message is None else message
        delta = datetime.timedelta(seconds=elapsed)
        print(f'[{self._name}] {message} elapsed [{str(delta)}]')
        return self

    def reset(self) -> 'Timer':
        """Reset the timer"""
        self._start_time = None
        return self


def batch_eval_recall(model: torch.nn.Module,
                     recall_k: List[int],
                     eval_data,
                     R: torch.Tensor,
                     device: torch.device) -> Tuple[List[float], List[float], List[float], np.ndarray]:
    """
    Evaluate model performance with batch processing
    
    Args:
        model: PyTorch model
        recall_k: List of K values for recall@K
        eval_data: Evaluation dataset
        R: Rating matrix
        device: PyTorch device
    
    Returns:
        Tuple of (recall, precision, ndcg, predictions)
    """
    # Set top_k for evaluation
    model.top_k = max(recall_k)
    
    # Prepare IDCG values
    idcg_array = 1 / np.log2(np.arange(recall_k[-1]) + 2)
    idcg_table = np.zeros(recall_k[-1])
    for i in range(recall_k[-1]):
        idcg_table[i] = np.sum(idcg_array[:(i + 1)])

    # Get predictions batch by batch
    tf_eval_preds_batch = []
    model.eval()
    with torch.no_grad():
        for eval_start, eval_stop in eval_data.eval_batch:
            R_input = R[eval_data.test_item_ids].to(device)
            user_input = torch.tensor(eval_data.test_user_ids[eval_start:eval_stop], device=device)
            
            tf_eval_preds = model(R_input, is_training=False, user_input=user_input)
            tf_eval_preds_batch.append(tf_eval_preds.cpu().numpy())

    tf_eval_preds = np.concatenate(tf_eval_preds_batch)
    preds_all = tf_eval_preds

    recall = []
    precision = []
    ndcg = []

    # Calculate metrics for each K
    for at_k in recall_k:
        preds_k = preds_all[:, :at_k]
        k = preds_k.shape[1]
        y = eval_data.R_test_inf

        # Create sparse matrix for predictions
        x = scipy.sparse.coo_matrix(
            (np.ones_like(preds_k).reshape(-1),
             (np.repeat(np.arange(y.shape[0]), k), preds_k.reshape(-1))),
            shape=y.shape
        )

        z = y.multiply(x)

        # Calculate recall
        recall_users = np.divide(np.sum(z, 1), np.sum(y, 1))
        recall.append(float(np.mean(recall_users)))

        # Calculate precision
        precision_users = np.sum(z, 1) / k
        precision.append(float(np.mean(precision_users)))

        # Calculate NDCG
        rows = x.row
        cols = x.col
        y_csr = y.tocsr()
        dcg_array = y_csr[(rows, cols)].A1.reshape((preds_k.shape[0], -1))
        dcg = np.sum(dcg_array * idcg_array[:k].reshape((1, -1)), axis=1)
        
        idcg = np.sum(y, axis=1) - 1
        idcg[np.where(idcg >= k)] = k - 1
        idcg = idcg_table[idcg.astype(int)]

        ndcg_users = dcg.reshape([-1]) / idcg.reshape([-1])
        ndcg.append(float(np.mean(ndcg_users)))

    return recall, precision, ndcg, preds_all


class BiasEvaluator:
    """Evaluator for analyzing recommendation bias"""
    def __init__(self, data_name: str, test_eval, old_cold_idx: np.ndarray):
        # Load necessary data
        user_cold_test_like = list(np.load(f'../Data/{data_name}/user_cold_test_like.npy', allow_pickle=True))
        item_AS_list_all = np.load(f'../Data/{data_name}/item_audience_size_list.npy')
        
        item_old2new_id_dict = test_eval.test_item_ids_map
        user_old2new_id_dict = test_eval.test_user_ids_map

        self.num_user = len(user_old2new_id_dict)
        self.num_item = len(item_old2new_id_dict)

        # Process item audience size list
        item_AS_list = np.zeros(len(item_old2new_id_dict)).astype(np.float32)
        for i in range(len(item_AS_list_all)):
            if i in item_old2new_id_dict:
                item_AS_list[item_old2new_id_dict[i]] = item_AS_list_all[i]

        # Convert old cold indices to new indices
        itemIdsNew = copy.copy(old_cold_idx)
        for i in range(len(old_cold_idx)):
            itemIdsNew[i] = item_old2new_id_dict[old_cold_idx[i]]
        self.cold_idx = itemIdsNew

        # Process user cold test likes
        self.cold_test_like = [[] for _ in range(len(user_old2new_id_dict))]
        for old_uid in range(len(user_cold_test_like)):
            if old_uid in user_old2new_id_dict:
                old_test_like = user_cold_test_like[old_uid]
                test_like = []
                for old_iid in old_test_like:
                    if old_iid in item_old2new_id_dict:
                        test_like.append(item_old2new_id_dict[old_iid])
                self.cold_test_like[user_old2new_id_dict[old_uid]] = np.array(test_like).astype(int)

        self.item_cold_pop = item_AS_list[self.cold_idx]

    def bias_analysis(self, rank_matrix: np.ndarray, k: int = 100) -> None:
        """
        Analyze bias in recommendations
        
        Args:
            rank_matrix: Matrix of ranked items
            k: Number of top items to consider
        """
        item_attention_count = np.zeros(self.num_item)
        item_count = np.zeros(self.num_item)

        # Calculate attention scores
        for u in range(self.num_user):
            u_rank_list = rank_matrix[u]
            u_cold_like_set = set(self.cold_test_like[u])

            match_item_set = set()
            for rank, iid in enumerate(u_rank_list):
                if rank == k:
                    break
                if iid in u_cold_like_set:
                    item_attention_count[iid] += (1. / np.log2(rank + 2))
                    item_count[iid] += 1.
                    match_item_set.add(iid)

            unmatch_item_set = u_cold_like_set - match_item_set
            for iid in unmatch_item_set:
                item_count[iid] += 1.

        # Calculate average attention and analyze distribution
        item_avg_attention = item_attention_count / (item_count + 1e-7)
        item_cold_avg_attention = item_avg_attention[self.cold_idx]

        # Calculate metrics for different percentiles
        minority_sizes = {
            '10%': int(len(item_cold_avg_attention) * 0.1),
            '20%': int(len(item_cold_avg_attention) * 0.2),
            '30%': int(len(item_cold_avg_attention) * 0.3),
            '50%': int(len(item_cold_avg_attention) * 0.5)
        }

        minority_indices = {
            size: np.argpartition(item_cold_avg_attention, n)[:n]
            for size, n in minority_sizes.items()
        }

        majority_10_idx = np.argpartition(item_cold_avg_attention, -minority_sizes['10%'])[-minority_sizes['10%']:]

        # Print results
        print('=' * 60)
        print(f'PCC attention for cold = {stats.pearsonr(item_cold_avg_attention + 1e-7, self.item_cold_pop + 1e-7)[0]:.4f}')
        print(f'Average attention for cold = {np.mean(item_cold_avg_attention):.4f}')
        
        for size, idx in minority_indices.items():
            print(f'Minority {size} attention = {np.mean(item_cold_avg_attention[idx]):.4f}')
            
        print('=' * 60)
        print(f'Majority 10% attention = {np.mean(item_cold_avg_attention[majority_10_idx]):.4f}')
        print('=' * 60)


def negative_sampling(pos_user_array: np.ndarray,
                     pos_item_array: np.ndarray,
                     neg: int,
                     item_warm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform negative sampling for training
    
    Args:
        pos_user_array: Array of positive user indices
        pos_item_array: Array of positive item indices
        neg: Number of negative samples per positive sample
        item_warm: Array of warm item indices
    
    Returns:
        Tuple of (user indices, item indices, targets)
    """
    user_pos = pos_user_array.reshape((-1))
    user_neg = np.tile(pos_user_array, neg).reshape((-1))
    pos = pos_item_array.reshape((-1))
    neg = np.random.choice(item_warm, size=(neg * pos_user_array.shape[0]), replace=True).reshape((-1))
    
    target_pos = np.ones_like(pos)
    target_neg = np.zeros_like(neg)
    
    return (np.concatenate((user_pos, user_neg)),
            np.concatenate((pos, neg)),
            np.concatenate((target_pos, target_neg)))