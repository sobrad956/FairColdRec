import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
import scipy.sparse
from dataclasses import dataclass


@dataclass
class MovieLensData:
    """Container for MovieLens dataset splits and metadata"""
    train_data: pd.DataFrame
    valid_data: pd.DataFrame
    test_data: pd.DataFrame
    n_users: int
    n_items: int
    user2idx: Dict[int, int]
    item2idx: Dict[int, int]
    item_content: Optional[np.ndarray] = None  # For genres
    user_content: Optional[np.ndarray] = None  # For demographics


class MovieLensDataset(Dataset):
    """PyTorch Dataset for MovieLens"""
    def __init__(self, ratings: pd.DataFrame):
        self.users = torch.tensor(ratings['user_idx'].values, dtype=torch.long)
        self.items = torch.tensor(ratings['item_idx'].values, dtype=torch.long)
        self.ratings = torch.tensor(ratings['rating'].values, dtype=torch.float)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.items[idx], self.ratings[idx]


def load_movielens(data_path: str = 'MovieLens1M',
                  valid_ratio: float = 0.1,
                  test_ratio: float = 0.2,
                  cold_start: bool = False,
                  random_state: int = 42) -> MovieLensData:
    """
    Load and preprocess MovieLens 1M dataset
    
    Args:
        data_path: Path to data directory
        valid_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        cold_start: If True, ensures validation and test items are cold-start
        random_state: Random seed for reproducibility
    """
    # Load ratings
    ratings_file = Path(data_path) / 'ratings.dat'
    ratings = pd.read_csv(
        ratings_file, 
        sep='::', 
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        engine='python'
    )
    
    # Create user and item mappings
    user2idx = {id_: idx for idx, id_ in enumerate(ratings['user_id'].unique())}
    item2idx = {id_: idx for idx, id_ in enumerate(ratings['movie_id'].unique())}
    
    # Convert IDs to indices
    ratings['user_idx'] = ratings['user_id'].map(user2idx)
    ratings['item_idx'] = ratings['movie_id'].map(item2idx)
    
    if cold_start:
        # Implement cold-start splitting
        train_data, valid_data, test_data = cold_start_split(
            ratings,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            random_state=random_state
        )
    else:
        # Original temporal splitting
        ratings = ratings.sort_values('timestamp')
        
        # First split off test set
        train_valid_data, test_data = train_test_split(
            ratings,
            test_size=test_ratio,
            shuffle=False
        )
        
        # Then split remaining data into train and validation
        train_data, valid_data = train_test_split(
            train_valid_data,
            test_size=valid_ratio/(1-test_ratio),
            shuffle=False
        )

    # Load and process content features
    movies_file = Path(data_path) / 'movies.dat'
    movies = pd.read_csv(
        movies_file,
        sep='::',
        names=['movie_id', 'title', 'genres'],
        engine='python',
        encoding='latin-1'
    )
    
    # Process genres into one-hot encoding
    genres = movies['genres'].str.get_dummies(sep='|')
    item_content = np.zeros((len(item2idx), len(genres.columns)))
    for movie_id in movies['movie_id']:
        if movie_id in item2idx:
            idx = item2idx[movie_id]
            item_content[idx] = genres.loc[movies['movie_id'] == movie_id].values

    # Print dataset statistics
    print(f"Dataset loaded with cold_start={cold_start}:")
    print(f"Train: {len(train_data)} interactions")
    print(f"Valid: {len(valid_data)} interactions")
    print(f"Test: {len(test_data)} interactions")
    
    if cold_start:
        train_items = set(train_data['item_idx'])
        valid_items = set(valid_data['item_idx'])
        test_items = set(test_data['item_idx'])
        print(f"Cold-start statistics:")
        print(f"Valid items not in train: {len(valid_items - train_items)}")
        print(f"Test items not in train: {len(test_items - train_items)}")
    
    return MovieLensData(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        n_users=len(user2idx),
        n_items=len(item2idx),
        user2idx=user2idx,
        item2idx=item2idx,
        item_content=item_content
    )

def cold_start_split(ratings: pd.DataFrame,
                    valid_ratio: float = 0.1,
                    test_ratio: float = 0.2,
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split ratings ensuring validation and test items are cold-start
    
    Args:
        ratings: DataFrame with rating data
        valid_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        random_state: Random seed
    
    Returns:
        Tuple of (train_data, valid_data, test_data)
    """
    np.random.seed(random_state)
    
    # Get all unique items
    all_items = ratings['item_idx'].unique()
    n_items = len(all_items)
    
    # Calculate number of items for valid and test
    n_test = int(n_items * test_ratio)
    n_valid = int(n_items * valid_ratio)
    
    # Randomly select items for valid and test sets
    test_items = set(np.random.choice(all_items, n_test, replace=False))
    remaining_items = list(set(all_items) - test_items)
    valid_items = set(np.random.choice(remaining_items, n_valid, replace=False))
    train_items = set(remaining_items) - valid_items
    
    # Split the data
    train_data = ratings[ratings['item_idx'].isin(train_items)]
    valid_data = ratings[ratings['item_idx'].isin(valid_items)]
    test_data = ratings[ratings['item_idx'].isin(test_items)]
    
    # Sort by timestamp within each split
    train_data = train_data.sort_values('timestamp')
    valid_data = valid_data.sort_values('timestamp')
    test_data = test_data.sort_values('timestamp')
    
    return train_data, valid_data, test_data


def create_sparse_matrix(data: pd.DataFrame,
                        n_users: int,
                        n_items: int) -> scipy.sparse.csr_matrix:
    """Create sparse rating matrix from DataFrame"""
    return scipy.sparse.coo_matrix(
        (data['rating'], (data['user_idx'], data['item_idx'])),
        shape=(n_users, n_items)
    ).tocsr()


def get_data_loaders(ml_data: MovieLensData,
                     batch_size: int = 1024,
                     num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train/valid/test sets"""
    train_dataset = MovieLensDataset(ml_data.train_data)
    valid_dataset = MovieLensDataset(ml_data.valid_data)
    test_dataset = MovieLensDataset(ml_data.test_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader


def prepare_ml_pipeline(data_path: str = 'MovieLens1M',
                       batch_size: int = 1024,
                       cold_start: bool = False) -> Tuple[MovieLensData, DataLoader, DataLoader, DataLoader]:
    """
    Prepare complete MovieLens pipeline
    
    Returns:
        Tuple of (data container, train loader, valid loader, test loader)
    """
    # Load and process data
    ml_data = load_movielens(data_path, cold_start=cold_start)
    
    # Create data loaders
    train_loader, valid_loader, test_loader = get_data_loaders(
        ml_data, batch_size=batch_size
    )
    
    return ml_data, train_loader, valid_loader, test_loader