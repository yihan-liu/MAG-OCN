# train.py

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from randomizer import *
from preprocessor import OCNMoleculeDataset
from model import OCNTransformer
from utils import *

def main(args):
    # Define the augmentations for the dataset.
    augmentations = [
        OCNRandomTranslation(2.0),
        OCNRandomRotation(),
        OCNRandomReflection(),
        OCNRandomMicroPerturbation(position_noise=0.01, moment_noise=0.01)
    ]

    # Create the traindataset.
    train_filenames = [fn + '.csv' for fn in args.train_filenames]
    dataset = OCNMoleculeDataset(
        root=args.root,
        filenames=train_filenames,
        dataset_size=args.num_samples,
        threshold=2.0,
        num_atoms_in_sample=args.num_atoms_in_sample,
        augmentations=augmentations,
        processed_dir=args.processed_dir
    )

    # Build test dataset from test_filenames (unknown datasets).
    test_filenames = [fn + '.csv' for fn in args.test_filenames]
    test_dataset = OCNMoleculeDataset(
        root=args.root,
        filenames=test_filenames,
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

    # Print dataset shapes
    print(f"Total dataset size: {len(dataset)}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(validate_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Define model
    model = OCNTransformer(n_heads=args.n_heads, num_layers=args.num_layers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(sum(p.numel() for p in model.parameters()))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(model)

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

    # Filenames
    parser.add_argument('--train-filenames', nargs='+', help='List of csv files for training/validation.')
    parser.add_argument('--test-filenames', nargs='+', help='List of csv files for testing (unseen data).')
    parser.add_argument('-r', '--root', default='./raw', help='Root folder for the data.')
    parser.add_argument('--processed-dir', default='./processed', help='Directory for processed data.')

    # Sample size
    parser.add_argument('-n', '--num-samples', type=int, default=1000, help='Number of samples to generate.')
    parser.add_argument('-a', '--num-atoms-in-sample', type=int, default=16, help='Number of atoms in a sample.')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')

    # Model parameters
    parser.add_argument('--n-heads', type=int, default=4, help='Number of heads in self attention.')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of transformer blocks.')

    args = parser.parse_args()

    main(args)