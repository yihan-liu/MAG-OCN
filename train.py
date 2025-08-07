# train.py
# Modified training protocol:
# - Training datasets: All files ending with '_expanded.csv'
# - Test datasets: All other CSV files in the raw directory

import argparse
import os
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from prettytable import PrettyTable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_util.randomizer import *
from data_util.preprocessor import OCNMoleculeDataset
from model.ocn_transformer import OCNTransformer
from utils import *

def print_model_info(model, args, train_size, val_size, test_size):
    """Print model and training setup information in a formatted table."""
    
    # Model architecture summary table
    arch_table = PrettyTable()
    arch_table.field_names = ["Component", "Value"]
    arch_table.add_row(["Model Type", "OCN Transformer"])
    arch_table.add_row(["Number of Heads", args.n_heads])
    arch_table.add_row(["Number of Layers", args.num_layers])
    arch_table.add_row(["Total Parameters", f"{sum(p.numel() for p in model.parameters()):,}"])
    arch_table.add_row(["Device", next(model.parameters()).device])
    
    # Detailed layer-by-layer information
    layer_table = PrettyTable()
    layer_table.field_names = ["Layer/Block", "Type", "Input Shape", "Output Shape", "Parameters"]
    
    def format_shape(shape):
        """Format tensor shape for display"""
        if isinstance(shape, (list, tuple)):
            return f"[{', '.join(map(str, shape))}]"
        return str(shape)
    
    def count_parameters(module):
        """Count parameters in a module"""
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    # Input projection layer
    input_proj = model.input_projection
    layer_table.add_row([
        "Input Projection",
        "Linear",
        "[B, N, 6]",
        f"[B, N, {input_proj.out_features}]",
        f"{count_parameters(input_proj):,}"
    ])
    
    # Transformer blocks
    for i, block in enumerate(model.transformer_blocks):
        block_name = f"Transformer Block {i+1}"
        d_model = block.attn.d_model
        n_heads = block.attn.n_heads
        
        # Self-attention layer
        layer_table.add_row([
            f"  ├─ Self-Attention",
            f"MultiHead (h={n_heads})",
            f"[B, N, {d_model}]",
            f"[B, N, {d_model}]",
            f"{count_parameters(block.attn):,}"
        ])
        
        # Layer norm 1
        layer_table.add_row([
            f"  ├─ LayerNorm 1",
            "LayerNorm",
            f"[B, N, {d_model}]",
            f"[B, N, {d_model}]",
            f"{count_parameters(block.norm1):,}"
        ])
        
        # Feed-forward network
        ffn_layers = list(block.ffn.children())
        ffn_params = count_parameters(block.ffn)
        layer_table.add_row([
            f"  ├─ Feed-Forward",
            f"Linear→ReLU→Linear",
            f"[B, N, {d_model}]",
            f"[B, N, {d_model}]",
            f"{ffn_params:,}"
        ])
        
        # Layer norm 2
        layer_table.add_row([
            f"  └─ LayerNorm 2",
            "LayerNorm",
            f"[B, N, {d_model}]",
            f"[B, N, {d_model}]",
            f"{count_parameters(block.norm2):,}"
        ])
        
        # Block summary
        block_params = count_parameters(block)
        layer_table.add_row([
            f"{block_name} Total",
            "Transformer Block",
            f"[B, N, {d_model}]",
            f"[B, N, {d_model}]",
            f"{block_params:,}"
        ])
        layer_table.add_row(["", "", "", "", ""])  # Empty row for spacing
    
    # Output layer
    out_layer = model.out_layer
    layer_table.add_row([
        "Output Layer",
        "Linear",
        f"[B, N, {out_layer.in_features}]",
        "[B, N, 1]",
        f"{count_parameters(out_layer):,}"
    ])
    
    # Training setup table
    setup_table = PrettyTable()
    setup_table.field_names = ["Parameter", "Value"]
    setup_table.add_row(["Batch Size", args.batch_size])
    setup_table.add_row(["Learning Rate", args.lr])
    setup_table.add_row(["Epochs", args.epochs])
    setup_table.add_row(["Num Atoms per Sample", args.num_atoms_in_sample])
    setup_table.add_row(["Num Samples", args.num_samples])
    
    # Dataset size table
    data_table = PrettyTable()
    data_table.field_names = ["Dataset Split", "Size"]
    data_table.add_row(["Training", train_size])
    data_table.add_row(["Validation", val_size])
    data_table.add_row(["Test", test_size])
    
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*70)
    print(arch_table)
    
    print("\n" + "="*70)
    print("DETAILED LAYER INFORMATION")
    print("="*70)
    print("Legend: B=Batch Size, N=Number of Atoms")
    print(layer_table)
    
    print("\n" + "="*70)
    print("TRAINING SETUP")
    print("="*70)
    print(setup_table)
    
    print("\n" + "="*70)
    print("DATASET INFORMATION")
    print("="*70)
    print(data_table)
    print("="*70 + "\n")

def main(args):
    # Automatically detect training and test files based on naming convention
    # Training files: those ending with '_expanded.csv'
    # Test files: all other csv files
    
    all_csv_files = glob.glob(os.path.join(args.root, '*.csv'))
    
    # Extract just the filenames without extension for the dataset loader
    train_filenames = []
    test_filenames = []
    
    for file_path in all_csv_files:
        filename = os.path.basename(file_path)
        filename_without_ext = filename[:-4]  # Remove '.csv'
        
        if filename.endswith('_expanded.csv'):
            train_filenames.append(filename_without_ext)
        else:
            test_filenames.append(filename_without_ext)
    
    print(f"Training files detected: {train_filenames}")
    print(f"Test files detected: {test_filenames}")
    
    # Define the augmentations for the dataset.
    augmentations = [
        OCNRandomTranslation(2.0),
        OCNRandomRotation(),
        OCNRandomReflection(),
        OCNRandomMicroPerturbation(position_noise=0.01, moment_noise=0.01)
    ]

    # Create the training dataset from expanded files.
    train_csv_filenames = [fn + '.csv' for fn in train_filenames]
    dataset = OCNMoleculeDataset(
        root=args.root,
        filenames=train_csv_filenames,
        dataset_size=args.num_samples,
        threshold=2.0,
        num_atoms_in_sample=args.num_atoms_in_sample,
        augmentations=augmentations,
        processed_dir=args.processed_dir
    )

    # Build test dataset from non-expanded files.
    test_csv_filenames = [fn + '.csv' for fn in test_filenames]
    test_dataset = OCNMoleculeDataset(
        root=args.root,
        filenames=test_csv_filenames,
        dataset_size=int(args.num_samples // 10),
        threshold=2.0,
        num_atoms_in_sample=args.num_atoms_in_sample,
        augmentations=None,
        processed_dir=args.processed_dir
    )

    # Split dataset into train and validation
    train_ratio = 0.8
    train_size = int(train_ratio * len(dataset))
    validate_size = len(dataset) - train_size
    train_dataset, validate_dataset = random_split(dataset, [train_size, validate_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False)

    # Test loader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) # Added batch_size and shuffle for consistency

    # Define model
    model = OCNTransformer(n_heads=args.n_heads, num_layers=args.num_layers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Add learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 10
    
    # Print formatted model and setup information
    print_model_info(model, args, len(train_dataset), len(validate_dataset), len(test_dataset))

    train_losses = []
    train_r2s = []
    val_losses = []
    val_r2s = []
    test_losses = []
    test_r2s = []

    # Training loop
    epoch_iterator = tqdm(range(1, args.epochs + 1), desc='Training epochs')
    for epoch in epoch_iterator:
        model.train()
        epoch_train_loss = 0.0
        epoch_train_r2 = 0.0
        total_train = 0

        # Lists to store predictions and targets for R^2 computation.
        train_preds = []
        train_targets = []

        for batch in train_loader:
            # Features shape: [B, num_atoms, 6]
            # First 3 columns: one-hot atom types
            # Last 3 columns: spatial coordinates
            features = batch['features'].to(device)
            atom_type = features[:, :, :3]                      # [B, num_atoms, 3]
            spatial_location = features[:, :, 3:]               # [B, num_atoms, 3]
            bond_influence = batch['bond_influence'].to(device) # [B, num_atoms, num_atoms]
            targets = batch['targets'].to(device)               # [B, num_atoms]
            
            optimizer.zero_grad()
            outputs = model(atom_type, spatial_location, bond_influence)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = features.size(0)
            epoch_train_loss += loss.item() * batch_size
            total_train += batch_size

            train_preds.append(outputs.detach().cpu().numpy())
            train_targets.append(targets.detach().cpu().numpy())

        # epoch train loss
        epoch_train_loss /= total_train

        # epoch train r2
        train_preds = np.concatenate(train_preds, axis=0).flatten()
        train_targets = np.concatenate(train_targets, axis=0).flatten()
        epoch_train_r2 = r2(train_targets,  train_preds)

        train_losses.append(epoch_train_loss)
        train_r2s.append(epoch_train_r2)

        # validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_r2 = 0.0
        total_val = 0

        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in validate_loader:
                features = batch['features'].to(device)
                atom_type = features[:, :, :3]
                spatial_location = features[:, :, 3:]
                bond_influence = batch['bond_influence'].to(device)
                targets = batch['targets'].to(device)
                outputs = model(atom_type, spatial_location, bond_influence)
                loss = criterion(outputs, targets)

                batch_size = features.size(0)
                epoch_val_loss += loss.item() * batch_size
                total_val += batch_size

                val_preds.append(outputs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())
        
        # epoch validate loss
        epoch_val_loss /= total_val
        
        # epoch validate r2
        val_preds = np.concatenate(val_preds, axis=0).flatten()
        val_targets = np.concatenate(val_targets, axis=0).flatten()
        epoch_val_r2 = r2(val_targets, val_preds)

        val_losses.append(epoch_val_loss)
        val_r2s.append(epoch_val_r2)

        # Update tqdm postfix for train and validation metrics
        epoch_iterator.set_postfix({
            'Train Loss': f'{epoch_train_loss:.4f}',
            'Train R2': f'{epoch_train_r2:.4f}',
            'Val Loss': f'{epoch_val_loss:.4f}',
            'Val R2': f'{epoch_val_r2:.4f}'
        })
        
        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

        # Every 10 epochs (and at the final epoch), print progress using tqdm.write
        if epoch % 10 == 0 or epoch == args.epochs:

            # test (from unknown dataset)
            epoch_test_loss = 0.0
            total_test = 0
            test_preds = []
            test_targets = []
            with torch.no_grad():
                for batch in test_loader:
                    features = batch['features'].to(device)
                    atom_type = features[:, :, :3]
                    spatial_location = features[:, :, 3:]
                    bond_influence = batch['bond_influence'].to(device)
                    targets = batch['targets'].to(device)
                    outputs = model(atom_type, spatial_location, bond_influence)
                    loss = criterion(outputs, targets)
                    
                    batch_size = features.size(0)
                    epoch_test_loss += loss.item() * batch_size
                    total_test += batch_size
                    
                    test_preds.append(outputs.cpu().numpy())
                    test_targets.append(targets.cpu().numpy())
            
            epoch_test_loss /= total_test
            test_preds = np.concatenate(test_preds, axis=0).flatten()
            test_targets = np.concatenate(test_targets, axis=0).flatten()
            epoch_test_r2 = r2(test_targets, test_preds)
            test_losses.append(epoch_test_loss)
            test_r2s.append(epoch_test_r2)

            tqdm.write(
                f"Epoch {epoch}/{args.epochs}: "
                f"Train Loss: {epoch_train_loss:.4f}, Train R2: {epoch_train_r2:.4f}, "
                f"Validate Loss: {epoch_val_loss:.4f}, Validate R2: {epoch_val_r2:.4f}, "
                f"Test Loss: {epoch_test_loss:.4f}, Test R2: {epoch_test_r2:.4f}"
            )

        # Step the scheduler based on validation loss
        scheduler.step(epoch_val_loss)

    # Save metric to a file
    np.savez('metrics/metrics.npz',
             train_losses=train_losses,
             train_r2s=train_r2s,
             val_losses=val_losses,
             val_r2s=val_r2s,
             test_losses=test_losses,
             test_r2s=test_r2s)
    print('Saved training metrics to metrics.npz.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data directories
    parser.add_argument('-r', '--root', default='./raw', help='Root folder for the data.')
    parser.add_argument('--processed-dir', default='./processed', help='Directory for processed data.')

    # Sample size
    parser.add_argument('-n', '--num-samples', type=int, default=1000, help='Number of samples to generate.')
    parser.add_argument('-a', '--num-atoms-in-sample', type=int, default=32, help='Number of atoms in a sample.')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')

    # Model parameters
    parser.add_argument('--n-heads', type=int, default=4, help='Number of heads in self attention.')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of transformer blocks.')

    args = parser.parse_args()

    main(args)