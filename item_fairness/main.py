import argparse
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import scipy.sparse
from tqdm import tqdm

from model import create_model
from data import load_eval_data, create_data_loaders
from utils import Timer, batch_eval_recall, BiasEvaluator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Debiasing Recommender")
    parser.add_argument('--data', type=str, default='ml1m',
                       help='path to eval in the downloaded folder')
    parser.add_argument('--alg', type=str, default='Heater',
                       help='algorithm')
    parser.add_argument('--model-select', nargs='+', type=int,
                       default=[100],
                       help='specify the fully-connected architecture')
    parser.add_argument('--eval-every', type=int, default=1,
                       help='evaluate every X epochs')
    parser.add_argument('--lr', type=float, default=0.005,
                       help='starting learning rate')
    parser.add_argument('--epoch', type=int, default=100,
                       help='number of epochs')
    parser.add_argument('--bs', type=int, default=50,
                       help='batch size')
    parser.add_argument('--eval-batch-size', type=int, default=40000,
                       help='evaluation batch size')
    parser.add_argument('--reg', type=float, default=0.00001,
                       help='regularization factor')
    parser.add_argument('--alpha', type=float, default=4,
                       help='alpha parameter')
    parser.add_argument('--device', type=str, default='cpu',
                       help='device to use (cuda/cpu)')
    return parser.parse_args()


def load_data(data_path: str, alg: str) -> Dict[str, Any]:
    """Load and prepare data"""
    timer = Timer(name='data_loading')
    data_dir = Path('../Data') / data_path
    
    # Load basic info
    with open(data_dir / 'info.pkl', 'rb') as f:
        info = pickle.load(f)
        n_users = info['num_user']
        n_items = info['num_item']

    # Load data files
    train_df = pd.read_csv(data_dir / 'train_df.csv', dtype=np.int32)
    cold_test_df = pd.read_csv(data_dir / 'cold_test_df.csv', dtype=np.int32)
    cold_vali_df = pd.read_csv(data_dir / 'cold_vali_df.csv', dtype=np.int32)

    # Get warm items
    item_warm = train_df['iid'].unique()

    # Load or compute rating matrix
    if alg == 'KNN':
        R = np.load(data_dir / 'KNN_R.npy')
    else:
        u_pref = np.load(data_dir / f'U_emb_{alg}.npy')
        v_pref = np.load(data_dir / f'I_emb_{alg}.npy')
        R = np.matmul(u_pref, v_pref.T)

    # Create sparse training matrix
    R_train = scipy.sparse.coo_matrix(
        (np.ones(len(train_df)),
         (train_df['iid'].values, train_df['uid'].values)),
        shape=(n_items, n_users)
    ).tolil()

    # Prepare evaluation data
    cold_test_eval = load_eval_data(
        cold_test_df.values.ravel().view(dtype=[('uid', np.int32), ('iid', np.int32)]),
        name='cold_test_eval'
    )
    cold_vali_eval = load_eval_data(
        cold_vali_df.values.ravel().view(dtype=[('uid', np.int32), ('iid', np.int32)]),
        name='cold_vali_eval'
    )

    # Create bias evaluator
    bias_evaluator = BiasEvaluator(
        data_path,
        cold_test_eval,
        np.unique(cold_test_df['iid'].values)
    )

    return {
        'num_user': n_users,
        'num_item': n_items,
        'item_warm': item_warm,
        'R': R,
        'R_train': R_train,
        'cold_test_eval': cold_test_eval,
        'cold_vali_eval': cold_vali_eval,
        'bias_evaluator': bias_evaluator
    }


def train(args: argparse.Namespace, data: Dict[str, Any]) -> None:
    """Main training loop"""
    device = torch.device(args.device)
    timer = Timer(name='training')
    recall_at = [15, 30, 200]

    # Create model
    model = create_model(
        args.model_select,
        data['num_user'],
        data['num_item'],
        args.reg
    ).to(device)

    # Prepare data
    R = torch.tensor(data['R'].T, device=device)
    mask = torch.tensor(data['R_train'].toarray(), device=device)
    R_output = preprocess_ratings(R, mask, data['item_warm'], args.alpha)

    # Create optimizer and scheduler
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.9)

    # Create data loader
    train_loader = create_data_loaders(
        R, R_output, data['item_warm'],
        args.bs
    )

    # Training loop
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        recon_loss = 0
        reg_loss = 0

        # Training step
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epoch}') as pbar:
            for R_batch, R_target in pbar:
                R_batch = R_batch.to(device)
                R_target = R_target.to(device)

                # Forward pass
                output = model.train_step(R_batch, R_target, optimizer)

                # Update metrics
                total_loss += output.loss_all.item()
                recon_loss += output.loss_r.item()
                reg_loss += output.reg_loss.item()

                pbar.set_postfix({
                    'loss': total_loss / (pbar.n + 1),
                    'recon': recon_loss / (pbar.n + 1),
                    'reg': reg_loss / (pbar.n + 1)
                })

        # Update learning rate
        scheduler.step()

        # Evaluation step
        if (epoch + 1) % args.eval_every == 0:
            evaluate_and_save(
                model, data, device, recall_at,
                args.data, args.alg, epoch,
                total_loss, recon_loss, reg_loss
            )


def preprocess_ratings(R: torch.Tensor,
                      mask: torch.Tensor,
                      item_warm: np.ndarray,
                      alpha: float) -> torch.Tensor:
    """
    Preprocess and rescale ratings
    
    Args:
        R: Rating matrix (n_items x n_users)
        mask: Training mask (n_items x n_users)
        item_warm: Warm item indices
        alpha: Scaling factor
    
    Returns:
        Preprocessed rating matrix
    """
    # Ensure proper dimensions
    R = R - R.min()
    R_output = R.clone()

    # Calculate position-based statistics
    pos_sum = mask.sum(dim=1, keepdim=True)  # (n_items x 1)
    pos_mean = torch.zeros_like(pos_sum)      # (n_items x 1)

    # Create boolean mask for warm items
    warm_mask = torch.zeros(R.shape[0], dtype=torch.bool, device=R.device)
    warm_mask[item_warm] = True
    warm_mask = warm_mask.unsqueeze(1)  # (n_items x 1)

    # Calculate means for warm items only
    item_mask = warm_mask.expand_as(mask)  # Expand to full size (n_items x n_users)
    masked_R = R_output * mask
    pos_mean[warm_mask.squeeze(1)] = (masked_R[item_mask].view(len(item_warm), -1).sum(dim=1, keepdim=True) / 
                                     pos_sum[warm_mask.squeeze(1)])

    # Apply alpha power and calculate weights
    pos_mean = pos_mean.pow(alpha)
    weights = torch.zeros_like(pos_sum)
    
    # Calculate weights for warm items
    warm_pos_mean = pos_mean[warm_mask.squeeze(1)]
    max_pos_mean = warm_pos_mean.max()
    weights[warm_mask.squeeze(1)] = max_pos_mean / warm_pos_mean

    # Apply weights to get final output
    R_output = R_output * weights * mask + (1 - mask) * R_output

    return R_output



def evaluate_and_save(model: torch.nn.Module,
                     data: Dict[str, Any],
                     device: torch.device,
                     recall_at: List[int],
                     data_path: str,
                     alg: str,
                     epoch: int,
                     total_loss: float,
                     recon_loss: float,
                     reg_loss: float) -> None:
    """
    Evaluate model and save results
    
    Args:
        model: PyTorch model
        data: Data dictionary
        device: PyTorch device
        recall_at: List of K values for recall@K
        data_path: Path to data directory
        alg: Algorithm name
        epoch: Current epoch
        total_loss: Total training loss
        recon_loss: Reconstruction loss
        reg_loss: Regularization loss
    """
    model.eval()
    timer = Timer(name='evaluation')

    # Evaluate on test set
    R = torch.tensor(data['R'].T, device=device)
    cold_test_recall, cold_test_precision, cold_test_ndcg, rank_matrix = batch_eval_recall(
        model,
        recall_at,
        data['cold_test_eval'],
        R,
        device
    )

    # Analyze bias
    data['bias_evaluator'].bias_analysis(rank_matrix)

    # Save results
    save_dir = Path('../Data') / data_path
    np.save(save_dir / f'item_new2old_list_Debias_scale_{alg}.npy',
            data['cold_test_eval'].test_item_new2old_list)
    np.save(save_dir / f'user_new2old_list_Debias_scale_{alg}.npy',
            data['cold_test_eval'].test_user_new2old_list)
    np.save(save_dir / f'rank_matrix_Debias_scale_{alg}.npy',
            rank_matrix)

    # Print results
    timer.toc(
        f'epoch={epoch} all_loss={total_loss:.4f} '
        f'r_loss={recon_loss:.4f} reg_loss={reg_loss:.4f}'
    )

    print('\t\t\t' + '\t '.join([f'@{i}'.ljust(6) for i in recall_at]))
    print(f'Test recall   \t{" ".join([f"{i:.6f}" for i in cold_test_recall])}')
    print(f'Test precision\t{" ".join([f"{i:.6f}" for i in cold_test_precision])}')
    print(f'Test ndcg     \t{" ".join([f"{i:.6f}" for i in cold_test_ndcg])}')
    print('!' * 150)


def main() -> None:
    """Main execution function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    # Print arguments
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    # Load data
    timer = Timer(name='main')
    timer.tic()
    data = load_data(args.data, args.alg)
    timer.toc('Data loaded')

    # Train model
    try:
        train(args, data)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()