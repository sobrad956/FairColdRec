import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.sparse
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import pandas as pd



@dataclass
class EvalData:
    """Data container for evaluation"""
    test_user_ids: np.ndarray
    test_item_ids: np.ndarray
    test_user_ids_map: Dict[int, int]
    test_item_ids_map: Dict[int, int]
    test_user_new2old_list: np.ndarray
    test_item_new2old_list: np.ndarray
    R_test_inf: scipy.sparse.csr_matrix
    R_train_inf: Optional[scipy.sparse.csr_matrix]
    eval_batch: List[Tuple[int, int]]

    @classmethod
    def from_test_data(cls, test_triplets: np.ndarray, 
                      train_data: Optional[np.ndarray] = None,
                      eval_batch_size: int = 100) -> 'EvalData':
        """
        Create EvalData instance from test triplets
        
        Args:
            test_triplets: Array of (user_id, item_id) pairs
            train_data: Optional training data array
            eval_batch_size: Batch size for evaluation
        """
        # Get unique user and item IDs
        test_item_ids = np.unique(test_triplets['iid'])
        test_user_ids = np.unique(test_triplets['uid'])

        # Create mapping dictionaries
        test_item_ids_map = {iid: i for i, iid in enumerate(test_item_ids)}
        test_user_ids_map = {uid: i for i, uid in enumerate(test_user_ids)}

        # Create new2old mapping arrays
        test_item_new2old_list = np.zeros(len(test_item_ids_map), dtype=np.int32)
        test_user_new2old_list = np.zeros(len(test_user_ids_map), dtype=np.int32)
        
        for old_id, new_id in test_item_ids_map.items():
            test_item_new2old_list[new_id] = old_id
        for old_id, new_id in test_user_ids_map.items():
            test_user_new2old_list[new_id] = old_id

        # Create sparse test matrix
        test_user_idx = [test_user_ids_map[t[0]] for t in test_triplets]
        test_item_idx = [test_item_ids_map[t[1]] for t in test_triplets]
        
        R_test_inf = scipy.sparse.coo_matrix(
            (np.ones(len(test_user_idx)),
             (test_user_idx, test_item_idx)),
            shape=[len(test_user_ids), len(test_item_ids)]
        ).tocsr()

        # Create sparse train matrix if train data provided
        if train_data is not None:
            train_indices = [(test_user_ids_map[t[0]], test_item_ids_map[t[1]])
                           for t in train_data
                           if t[1] in test_item_ids_map and t[0] in test_user_ids_map]
            
            if train_indices:
                train_users, train_items = zip(*train_indices)
                R_train_inf = scipy.sparse.coo_matrix(
                    (np.ones(len(train_indices)),
                     (train_users, train_items)),
                    shape=R_test_inf.shape
                ).tocsr()
            else:
                R_train_inf = None
        else:
            R_train_inf = None

        # Create evaluation batches
        eval_l = R_test_inf.shape[0]
        eval_batch = [(x, min(x + eval_batch_size, eval_l)) 
                     for x in range(0, eval_l, eval_batch_size)]

        return cls(
            test_user_ids=test_user_ids,
            test_item_ids=test_item_ids,
            test_user_ids_map=test_user_ids_map,
            test_item_ids_map=test_item_ids_map,
            test_user_new2old_list=test_user_new2old_list,
            test_item_new2old_list=test_item_new2old_list,
            R_test_inf=R_test_inf,
            R_train_inf=R_train_inf,
            eval_batch=eval_batch
        )

    def get_stats_string(self) -> str:
        """Get string representation of data statistics"""
        stats = [
            f'n_test_users: [{len(self.test_user_ids)}]',
            f'n_test_items: [{len(self.test_item_ids)}]'
        ]
        
        if self.R_train_inf is None:
            stats.append('R_train_inf: no R_train_inf for cold')
        else:
            stats.append(
                f'R_train_inf: shape={self.R_train_inf.shape} '
                f'nnz=[{len(self.R_train_inf.nonzero()[0])}]'
            )
            
        stats.append(
            f'R_test_inf: shape={self.R_test_inf.shape} '
            f'nnz=[{len(self.R_test_inf.nonzero()[0])}]'
        )
        
        return '\n\t'.join(stats)


class RecommendationDataset(Dataset):
    """PyTorch Dataset for recommendation data"""
    def __init__(self, 
                 R: torch.Tensor,
                 R_output: torch.Tensor,
                 item_indices: np.ndarray):
        """
        Args:
            R: Rating matrix
            R_output: Target rating matrix
            item_indices: Indices of items to use
        """
        self.R = R
        self.R_output = R_output
        self.item_indices = item_indices

    def __len__(self) -> int:
        return len(self.item_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item_idx = self.item_indices[idx]
        return self.R[item_idx], self.R_output[item_idx]


def load_eval_data(test_data: np.ndarray,
                  train_data: Optional[np.ndarray] = None,
                  eval_batch_size: int = 100,
                  name: str = "eval") -> EvalData:
    """
    Load and prepare evaluation data
    
    Args:
        test_data: Test data array
        train_data: Optional training data array
        eval_batch_size: Batch size for evaluation
        name: Name for logging purposes
    
    Returns:
        EvalData instance
    """
    eval_data = EvalData.from_test_data(
        test_data,
        train_data=train_data,
        eval_batch_size=eval_batch_size
    )
    
    print(f"Loaded {name}:")
    print(eval_data.get_stats_string())
    
    return eval_data


def create_data_loaders(R: torch.Tensor,
                       R_output: torch.Tensor,
                       item_warm: np.ndarray,
                       batch_size: int,
                       num_workers: int = 4) -> DataLoader:
    """
    Create PyTorch DataLoader for training
    
    Args:
        R: Rating matrix
        R_output: Target rating matrix
        item_warm: Indices of warm items
        batch_size: Batch size for training
        num_workers: Number of worker processes
    
    Returns:
        PyTorch DataLoader
    """
    dataset = RecommendationDataset(R, R_output, item_warm)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )