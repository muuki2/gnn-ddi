"""
Generate ground truth label files from OGB MolBACE dataset.

This script creates:
- test_labels.csv: Ground truth labels for the test set
- valid_labels.csv: Ground truth labels for the validation set

These files should be stored in a PRIVATE repository and NOT committed to the public repo.
"""

import os
import pandas as pd
import torch

# Fix for PyTorch 2.6+ compatibility with torch_geometric
import torch.serialization
try:
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
except ImportError:
    pass

from ogb.graphproppred import PygGraphPropPredDataset

def main():
    print("Loading OGB MolBACE dataset...")
    dataset = PygGraphPropPredDataset(name='ogbg-molbace', root='../data/ogb')
    
    # Get official splits
    split_idx = dataset.get_idx_split()
    
    # Create output directory
    output_dir = '../temp_private_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate TEST labels
    print("\nGenerating test labels...")
    test_idx = split_idx['test'].tolist()
    test_labels = [dataset[i].y.item() for i in test_idx]
    test_df = pd.DataFrame({'id': test_idx, 'target': test_labels})
    test_df.to_csv(os.path.join(output_dir, 'test_labels.csv'), index=False)
    print(f"  - Test set: {len(test_df)} molecules")
    print(f"  - Class distribution: {test_df['target'].value_counts().to_dict()}")
    
    # Generate VALIDATION labels
    print("\nGenerating validation labels...")
    valid_idx = split_idx['valid'].tolist()
    valid_labels = [dataset[i].y.item() for i in valid_idx]
    valid_df = pd.DataFrame({'id': valid_idx, 'target': valid_labels})
    valid_df.to_csv(os.path.join(output_dir, 'valid_labels.csv'), index=False)
    print(f"  - Validation set: {len(valid_df)} molecules")
    print(f"  - Class distribution: {valid_df['target'].value_counts().to_dict()}")
    
    # Also generate TRAIN labels (for reference, can be public)
    print("\nGenerating train labels (for reference)...")
    train_idx = split_idx['train'].tolist()
    train_labels = [dataset[i].y.item() for i in train_idx]
    train_df = pd.DataFrame({'id': train_idx, 'target': train_labels})
    train_df.to_csv(os.path.join(output_dir, 'train_labels.csv'), index=False)
    print(f"  - Train set: {len(train_df)} molecules")
    print(f"  - Class distribution: {train_df['target'].value_counts().to_dict()}")
    
    print(f"\n✅ All label files saved to: {os.path.abspath(output_dir)}")
    print("\nFiles created:")
    print(f"  - {output_dir}/test_labels.csv   (KEEP PRIVATE)")
    print(f"  - {output_dir}/valid_labels.csv  (KEEP PRIVATE)")
    print(f"  - {output_dir}/train_labels.csv  (Can be public)")
    
    print("\n⚠️  IMPORTANT: Copy test_labels.csv and valid_labels.csv to your private repo!")

if __name__ == "__main__":
    main()
