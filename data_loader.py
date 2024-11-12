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
                  random_state: int = 42) -> MovieLensData:
    """
    Load and preprocess MovieLens 1M dataset
    
    Args:
        data_path: Path to data directory
        valid_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        random_state: Random seed for reproducibility
    
    Returns:
        MovieLensData containing processed dataset
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
    
    # Sort by timestamp and get train/valid/test split
    ratings = ratings.sort_values('timestamp')
    
    # First split off test set
    train_valid_data, test_data = train_test_split(
        ratings,
        test_size=test_ratio,
        shuffle=False  # Keep temporal ordering
    )
    
    # Then split remaining data into train and validation
    train_data, valid_data = train_test_split(
        train_valid_data,
        test_size=valid_ratio/(1-test_ratio),
        shuffle=False  # Keep temporal ordering
    )
    
    # Load movie content (genres)
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
    
    # # Load user content (demographics)
    # users_file = Path(data_path) / 'users.dat'
    # users = pd.read_csv(
    #     users_file,
    #     sep='::',
    #     names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
    #     engine='python'
    # )
    
    # # Process user demographics
    # # Gender
    # users['gender'] = (users['gender'] == 'M').astype(int)
    
    # # Age groups (one-hot)
    # age_dummies = pd.get_dummies(users['age'], prefix='age')
    
    # # Occupation (one-hot)
    # occ_dummies = pd.get_dummies(users['occupation'], prefix='occ')
    
    # # Combine features
    # user_features = pd.concat([
    #     users[['gender']],
    #     age_dummies,
    #     occ_dummies
    # ], axis=1)
    
    # # Create user content matrix
    # user_content = np.zeros((len(user2idx), len(user_features.columns)))
    # for user_id in users['user_id']:
    #     if user_id in user2idx:
    #         idx = user2idx[user_id]
    #         user_content[idx] = user_features.loc[users['user_id'] == user_id].values

    print(f"Dataset loaded: {len(train_data)} train, {len(valid_data)} validation, {len(test_data)} test")
    print(f"Sparsity: {len(ratings)/(len(user2idx)*len(item2idx)):.5f}")
    
    return MovieLensData(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        n_users=len(user2idx),
        n_items=len(item2idx),
        user2idx=user2idx,
        item2idx=item2idx,
        item_content=item_content#,
        #user_content=user_content
    )


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
                       batch_size: int = 1024) -> Tuple[MovieLensData, DataLoader, DataLoader, DataLoader]:
    """
    Prepare complete MovieLens pipeline
    
    Returns:
        Tuple of (data container, train loader, valid loader, test loader)
    """
    # Load and process data
    ml_data = load_movielens(data_path)
    
    # Create data loaders
    train_loader, valid_loader, test_loader = get_data_loaders(
        ml_data, batch_size=batch_size
    )
    
    return ml_data, train_loader, valid_loader, test_loader