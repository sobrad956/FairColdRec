import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import torch.nn.functional as F
import torch
import torch.nn as nn
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

from data_loader import MovieLensData
from matrix_factor import BiasedMF
from heater import Heater, HeaterOutput
from debiasing import DebiasingModel


class RecommenderEvaluator:
    def __init__(self, ml_data: MovieLensData):
        self.ml_data = ml_data
        self.metrics = {}
        self.cold_start_items = self._identify_cold_start_items()
        
    def _identify_cold_start_items(self) -> set:
        """Identify cold start items (items only in test set)"""
        train_items = set(self.ml_data.train_data['item_idx'])
        test_items = set(self.ml_data.test_data['item_idx'])
        return test_items - train_items



    def evaluate_base_mf(self, 
                        model: BiasedMF,
                        data_loader: torch.utils.data.DataLoader,
                        k_values: List[int] = [5, 10, 20, 50],
                        device: str = 'cpu') -> Dict[str, float]:
        """
        Evaluate base matrix factorization model using NDCG@K
        
        Args:
            model: BiasedMF model
            data_loader: Test data loader
            k_values: List of K values for NDCG calculation
            device: Computing device
        
        Returns:
            Dictionary of NDCG scores for each K
        """
        model.eval()
        metrics = defaultdict(float)
        n_users = 0
        
        with torch.no_grad():
            for user_ids, item_ids, ratings in data_loader:
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                ratings = ratings.to(device)
                
                # Get predictions
                output = model(user_ids, item_ids)
                predictions = output.preds.cpu().numpy()
                true_ratings = ratings.cpu().numpy()
                
                # Calculate NDCG for each K
                ndcg_scores = self._calculate_ndcg(predictions, true_ratings, k_values)
                for k, score in ndcg_scores.items():
                    metrics[f'ndcg@{k}'] += score
                    
                n_users += 1
        
        # Average metrics
        for metric in metrics:
            metrics[metric] /= n_users
            
        self.metrics['base_mf'] = dict(metrics)
        return dict(metrics)
    
    def evaluate_heater(self,
                        heater: Heater,
                        base_model: BiasedMF,
                        k_values: List[int] = [5, 10, 20, 50],
                        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                        batch_size: int = 128) -> Dict[str, float]:
            
            heater.eval()
            metrics = defaultdict(float)
            n_users = 0
            
            with torch.no_grad():
                # Get base embeddings
                u_emb, i_emb = base_model.get_embeddings()
                
                # Process users in batches
                user_groups = list(self.ml_data.test_data.groupby('user_idx'))
                
                for batch_start in range(0, len(user_groups), batch_size):
                    batch_end = min(batch_start + batch_size, len(user_groups))
                    current_batch_size = batch_end - batch_start
                    
                    # Prepare batch data
                    batch_user_ids = []
                    batch_true_ratings = np.zeros((current_batch_size, len(i_emb)), dtype=np.float32)
                    
                    for i, (user_idx, group) in enumerate(user_groups[batch_start:batch_end]):
                        batch_user_ids.append(user_idx)
                        batch_true_ratings[i, group['item_idx'].values] = group['rating'].values
                    
                    # Convert to tensors
                    batch_u_emb = u_emb[batch_user_ids].to(device)
                    batch_true_ratings = torch.from_numpy(batch_true_ratings).to(device)
                    
                    try:
                        # Get HEATER predictions
                        output = heater(
                            batch_u_emb,
                            i_emb.to(device),
                            is_training=False
                        )
                        
                        # Apply softmax normalization to predictions
                        predictions = torch.softmax(output.preds, dim=1).cpu().numpy()
                        
                        # Calculate metrics for each user
                        for idx in range(current_batch_size):
                            user_preds = predictions[idx]
                            user_true = batch_true_ratings[idx].cpu().numpy()
                            
                            # Calculate NDCG for each K
                            ndcg_scores = self._calculate_ndcg(user_preds, user_true, k_values)
                            for k, score in ndcg_scores.items():
                                metrics[f'ndcg@{k}'] += score
                            
                            n_users += 1
                            
                        if (batch_start//batch_size) % 10 == 0:
                            print(f"Processed {n_users} users...")
                        
                    except Exception as e:
                        print(f"Error processing batch {batch_start//batch_size}: {str(e)}")
                        continue
                    
                    if n_users > 0 and n_users % 1000 == 0:
                        # Print intermediate metrics
                        print(f"\nIntermediate metrics after {n_users} users:")
                        curr_metrics = {k: v/n_users for k, v in metrics.items()}
                        for k in k_values:
                            print(f"NDCG@{k}: {curr_metrics[f'ndcg@{k}']:.4f}")
            
            # Average metrics
            for metric in metrics:
                metrics[metric] /= max(n_users, 1)
            
            self.metrics['heater'] = metrics
            return metrics
    
    def evaluate_debiased(self,
                        base_model: Union[BiasedMF, Heater],
                        debiasing_model: DebiasingModel,
                        k_values: List[int] = [5, 10, 20, 50],
                        device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, float]:
        """Evaluate debiased recommendations with proper user index mapping"""
        print("Evaluating debiased recommendations...")
        debiasing_model.eval()
        metrics = defaultdict(float)
        n_users = 0
        
        with torch.no_grad():
            # Get base embeddings
            if isinstance(base_model, BiasedMF):
                u_emb, i_emb = base_model.get_embeddings()
                R = torch.mm(i_emb, u_emb.t())
            else:  # Heater
                u_emb = base_model.user_embedding.weight
                i_emb = base_model.item_embedding.weight
                
                # Normalize embeddings
                u_norm = F.normalize(u_emb, p=2, dim=1)
                i_norm = F.normalize(i_emb, p=2, dim=1)
                
                # Calculate rating matrix
                R = torch.mm(i_norm, u_norm.t()) * 5.0
            
            R = R.to(device)
            print(f"Rating matrix shape: {R.shape}")
            
            # Create user index mapping
            unique_users = self.ml_data.train_data['user_idx'].unique()
            user_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(unique_users))}
            
            print(f"Number of unique users in training: {len(unique_users)}")
            print(f"Rating matrix users dimension: {R.shape[1]}")
            
            # Process test data by user
            for user_idx, group in self.ml_data.test_data.groupby('user_idx'):
                # Map the user index to the matrix dimension
                if user_idx not in user_mapping:
                    print(f"Skipping user {user_idx} - not in training data")
                    continue
                    
                mapped_user_idx = user_mapping[user_idx]
                if mapped_user_idx >= R.shape[1]:
                    print(f"Skipping user {user_idx} (mapped to {mapped_user_idx}) - out of bounds")
                    continue
                    
                # Get user's true ratings
                true_ratings = np.zeros(self.ml_data.n_items)
                item_indices = group['item_idx'].values
                valid_indices = item_indices[item_indices < self.ml_data.n_items]
                true_ratings[valid_indices] = group.loc[group['item_idx'].isin(valid_indices), 'rating'].values
                
                try:
                    # Get debiased predictions
                    user_R = R[:, mapped_user_idx:mapped_user_idx+1].T
                    debiased_output = debiasing_model(user_R, is_training=False)
                    predictions = debiased_output.preds.squeeze().cpu().numpy()
                    
                    # Ensure predictions have the right shape
                    if len(predictions.shape) == 0:
                        predictions = np.array([predictions])
                    elif len(predictions.shape) > 1:
                        predictions = predictions.flatten()
                    
                    # Pad or truncate predictions if necessary
                    if len(predictions) < len(true_ratings):
                        predictions = np.pad(predictions, (0, len(true_ratings) - len(predictions)))
                    elif len(predictions) > len(true_ratings):
                        predictions = predictions[:len(true_ratings)]
                    
                    # Print shapes for debugging (occasionally)
                    if n_users % 100 == 0:
                        print(f"\nUser {user_idx} (mapped to {mapped_user_idx}):")
                        print(f"Predictions shape: {predictions.shape}")
                        print(f"True ratings shape: {true_ratings.shape}")
                        print(f"Non-zero true ratings: {np.count_nonzero(true_ratings)}")
                    
                    # Calculate NDCG for each K
                    ndcg_scores = self._calculate_ndcg(predictions, true_ratings, k_values)
                    for k, score in ndcg_scores.items():
                        metrics[f'ndcg@{k}'] += score
                    
                    n_users += 1
                    
                except Exception as e:
                    print(f"Error processing user {user_idx}: {str(e)}")
                    continue
                
                # Print progress periodically
                if n_users % 100 == 0:
                    print(f"Processed {n_users} users...")
        
        # Average metrics
        if n_users > 0:
            for metric in metrics:
                metrics[metric] /= n_users
        
        print("\nDebiased Model Results:")
        print(f"Successfully processed {n_users} users")
        for k in k_values:
            print(f"NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
        
        # Store metrics for later comparison
        self.metrics['debiased'] = dict(metrics)
        return dict(metrics)


    def _calculate_ndcg(self, 
                    predictions: np.ndarray, 
                    true_ratings: np.ndarray,
                    k_values: List[int]) -> Dict[int, float]:
        """Calculate NDCG@K"""
        ndcg_scores = {}
        
        # Scale up predictions to prevent underflow
        predictions = predictions * 1e3  # Scale up by 1000
        
        for k in k_values:
            try:
                # Get top k predictions
                top_k_idx = np.argsort(predictions)[-k:][::-1]
                
                # Create binary relevance vectors
                pred_relevance = np.zeros_like(predictions)
                pred_relevance[top_k_idx] = 1
                
                # Calculate NDCG
                ndcg = ndcg_score(
                    true_ratings.reshape(1, -1),
                    pred_relevance.reshape(1, -1),
                    k=k
                )
                ndcg_scores[k] = ndcg
            except Exception as e:
                print(f"Warning: Error calculating NDCG@{k}: {str(e)}")
                ndcg_scores[k] = 0.0
                
        return ndcg_scores

    def _identify_cold_start_items(self) -> set:
        """Identify cold start items (items only in test set)"""
        train_items = set(self.ml_data.train_data['item_idx'])
        test_items = set(self.ml_data.test_data['item_idx'])
        cold_items = test_items - train_items
        print(f"Identified {len(cold_items)} cold-start items out of {len(test_items)} test items")
        return cold_items

    def analyze_popularity_bias(self, 
                              heater: Heater,
                              base_model: BiasedMF,
                              k: int = 20,
                              device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, float]:
        """
        Analyze popularity bias in recommendations
        
        Args:
            heater: Trained HEATER model
            base_model: Base MF model
            k: Number of recommendations to consider
            device: Computing device
        
        Returns:
            Dictionary of bias metrics
        """
        # Calculate item popularity from training data
        item_popularity = self.ml_data.train_data['item_idx'].value_counts()
        bias_metrics = {}
        
        # Get base embeddings
        u_emb, i_emb = base_model.get_embeddings()
        
        recommended_items = []
        for user_idx, group in self.ml_data.test_data.groupby('user_idx'):
            # Create dropout masks (use all information for evaluation)
            u_dropout_mask = torch.zeros(1, u_emb.shape[1], device=device)
            v_dropout_mask = torch.zeros(len(i_emb), i_emb.shape[1], device=device)
            
            try:
                # Get recommendations
                output = heater(
                    u_emb[user_idx:user_idx+1].to(device),
                    i_emb.to(device),
                    is_training=False,
                    u_content=torch.tensor(self.ml_data.user_content[user_idx:user_idx+1]).float().to(device) if self.ml_data.user_content is not None else None,
                    v_content=torch.tensor(self.ml_data.item_content).float().to(device) if self.ml_data.item_content is not None else None,
                    u_dropout_mask=u_dropout_mask,
                    v_dropout_mask=v_dropout_mask
                )
                
                # Get top-k recommendations
                _, top_items = torch.topk(output.preds, k=k)
                recommended_items.extend(top_items.cpu().numpy().flatten())
                
            except Exception as e:
                print(f"Error getting recommendations for user {user_idx}: {str(e)}")
                continue
        
        rec_popularity = pd.Series(recommended_items).value_counts()
        
        # Calculate Gini coefficient
        bias_metrics['gini_coefficient'] = self._calculate_gini(rec_popularity.values)
        
        # Calculate long-tail coverage
        long_tail = set(item_popularity[item_popularity < item_popularity.median()].index)
        bias_metrics['long_tail_coverage'] = len(set(rec_popularity.index) & long_tail) / len(long_tail)
        
        # Calculate popularity bias correlation
        common_items = set(rec_popularity.index) & set(item_popularity.index)
        if common_items:
            rec_pop = rec_popularity[list(common_items)]
            item_pop = item_popularity[list(common_items)]
            bias_metrics['popularity_correlation'] = rec_pop.corr(item_pop)
        
        return bias_metrics

    def _calculate_gini(self, array: np.ndarray) -> float:
        """Calculate Gini coefficient"""
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        return np.sum((2 * index - array.shape[0] - 1) * array) / (array.shape[0] * np.sum(array))

    def plot_performance_comparison(self, k_values: List[int], save_path: Optional[str] = None):
        """
        Plot performance comparison between base, HEATER, and debiased models
        
        Args:
            k_values: List of K values to plot
            save_path: Optional path to save the plot
        """
        # Check that we have all necessary metrics
        required_models = {'base_mf', 'heater', 'debiased'}
        if not required_models.issubset(self.metrics.keys()):
            missing = required_models - set(self.metrics.keys())
            raise ValueError(f"Missing metrics for models: {missing}")
        
        # Set up the plot with proper sizing
        plt.figure(figsize=(10, 6))
        
        # Set up bar positions
        x = np.arange(len(k_values))
        width = 0.25  # Width of bars
        
        # Get NDCG values for each model
        base_ndcg = [self.metrics['base_mf'][f'ndcg@{k}'] for k in k_values]
        heater_ndcg = [self.metrics['heater'][f'ndcg@{k}'] for k in k_values]
        debiased_ndcg = [self.metrics['debiased'][f'ndcg@{k}'] for k in k_values]
        
        # Plot bars
        plt.bar(x - width, base_ndcg, width, label='Base MF', color='#2ecc71')
        plt.bar(x, heater_ndcg, width, label='HEATER', color='#3498db')
        plt.bar(x + width, debiased_ndcg, width, label='Debiased', color='#e74c3c')
        
        # Customize plot
        plt.xlabel('K', fontsize=12)
        plt.ylabel('NDCG Score', fontsize=12)
        plt.title('NDCG@K Comparison', fontsize=14, pad=20)
        plt.xticks(x, [f'K={k}' for k in k_values])
        plt.legend(fontsize=10)
        
        # Add grid for better readability
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show plot
        plt.show()
        
        # Print numerical results
        print("\nNumerical Results:")
        print(f"{'K':<10} {'Base MF':<12} {'HEATER':<12} {'Debiased':<12}")
        print("-" * 46)
        for i, k in enumerate(k_values):
            print(f"{k:<10} {base_ndcg[i]:.4f}      {heater_ndcg[i]:.4f}      {debiased_ndcg[i]:.4f}")
