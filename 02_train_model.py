# train_model.py
import gc
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import polars as pl  # OPTIMIZATION: Use Polars instead of Pandas - faster and lower memory
from pathlib import Path
import glob
from tqdm import tqdm

DATA_DIR = Path("data")
BATCH_SIZE = 1024
EPOCHS = 5
INPUT_DIM = 30 + 10  # user + restaurant features


# DO NOT EDIT MODEL ARCHITECTURE
class RankNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)


def train():
    # ========================================================================
    # OPTIMIZATION 1: Load feature tables with Polars (faster than pandas)
    # Convert to numpy arrays for efficient indexing during training
    # ========================================================================
    print("\nLoading feature tables with Polars...")
    
    user_features_df = pl.read_parquet(DATA_DIR / "user_features.parquet")
    restaurant_features_df = pl.read_parquet(DATA_DIR / "restaurant_features.parquet")
    
    # Get feature columns (exclude id and location columns)
    user_feature_cols = [c for c in user_features_df.columns if c != 'user_id']
    restaurant_feature_cols = [c for c in restaurant_features_df.columns 
                               if c not in ['restaurant_id', 'latitude', 'longitude']]
    
    # Convert to numpy for fast indexing (Polars to_numpy is very efficient)
    user_features_np = user_features_df.select(user_feature_cols).to_numpy().astype(np.float32)
    restaurant_features_np = restaurant_features_df.select(restaurant_feature_cols).to_numpy().astype(np.float32)
    
    # Create ID to index mappings
    user_ids = user_features_df['user_id'].to_numpy()
    restaurant_ids = restaurant_features_df['restaurant_id'].to_numpy()
    user_id_to_idx = {int(uid): idx for idx, uid in enumerate(user_ids)}
    restaurant_id_to_idx = {int(rid): idx for idx, rid in enumerate(restaurant_ids)}
    
    # Free DataFrame memory
    del user_features_df, restaurant_features_df, user_ids, restaurant_ids
    gc.collect()
    
    print(f"  User features shape: {user_features_np.shape}")
    print(f"  Restaurant features shape: {restaurant_features_np.shape}")
    
    # ========================================================================
    # OPTIMIZATION 2: Get file list and calculate total samples
    # ========================================================================
    train_files = sorted(glob.glob(str(DATA_DIR / "train" / "*.parquet")))
    print(f"\nFound {len(train_files)} training files")
    
    # Count total samples using Polars (much faster than pandas)
    total_samples = 0
    file_lengths = []
    for f in train_files:
        # Polars scan_parquet with select is very fast for counting
        n = pl.scan_parquet(f).select(pl.len()).collect().item()
        file_lengths.append(n)
        total_samples += n
    
    print(f"Total training samples: {total_samples:,}")
    num_batches = sum(math.ceil(fl / BATCH_SIZE) for fl in file_lengths)
    print(f"Total batches per epoch: {num_batches:,}")
    
    # ========================================================================
    # Initialize model (same architecture - not changed)
    # ========================================================================
    model = RankNet(INPUT_DIM)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-8)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: 1e-8")
    
    epoch_times = []
    epoch_losses = []
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        # Process files one at a time (lazy loading pattern)
        with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{EPOCHS}") as pbar:
            for file_idx, train_file in enumerate(train_files):
                # ============================================================
                # OPTIMIZATION 3: Load and process file with Polars
                # Polars is 5-10x faster than pandas for parquet operations
                # ============================================================
                df = pl.read_parquet(train_file)
                
                # Get user and restaurant indices using numpy vectorized operations
                user_indices = np.array([user_id_to_idx[uid] for uid in df['user_id'].to_numpy()])
                rest_indices = np.array([restaurant_id_to_idx[rid] for rid in df['restaurant_id'].to_numpy()])
                
                # Concatenate features
                features = np.hstack([
                    user_features_np[user_indices],
                    restaurant_features_np[rest_indices]
                ])
                labels = df['click'].to_numpy().astype(np.float32)
                
                # Free Polars DataFrame
                del df
                
                # ============================================================
                # OPTIMIZATION 4: Process batches from this file
                # ============================================================
                n_samples = len(labels)
                for batch_start in range(0, n_samples, BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, n_samples)
                    
                    # OPTIMIZATION 5: Create tensors directly from numpy slices
                    x = torch.from_numpy(features[batch_start:batch_end])
                    y = torch.from_numpy(labels[batch_start:batch_end])
                    
                    # OPTIMIZATION 6: zero_grad with set_to_none for efficiency
                    optimizer.zero_grad(set_to_none=True)
                    
                    output = model(x)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    batch_count += 1
                    pbar.update(1)
                
                # Free memory after processing each file
                del features, labels, user_indices, rest_indices
                gc.collect()
        
        epoch_time = time.time() - start_time
        avg_loss = running_loss / batch_count
        
        epoch_times.append(epoch_time)
        epoch_losses.append(avg_loss)
        
        print(f"  Loss: {avg_loss:.4f}, Elapsed time: {epoch_time:.3f} seconds")
    
    # ========================================================================
    # Save model
    # ========================================================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    
    print(f"\nAverage epoch time: {np.mean(epoch_times):.3f} seconds")
    print(f"Final loss: {epoch_losses[-1]:.4f}")
    
    model_scripted = torch.jit.script(model)
    model_scripted.save(DATA_DIR / "model.pt")
    print(f"\nModel saved to {DATA_DIR / 'model.pt'}")


if __name__ == "__main__":
    train()
