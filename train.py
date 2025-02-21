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
    train_filenames = [fn + '.csv' for fn in args.train_filenames]

    # Define the augmentations for the dataset.
    augmentations = [
        OCNRandomTranslation(2.0),
        OCNRandomRotation(),
        OCNRandomReflection(),
        OCNRandomMicroPerturbation(position_noise=0.01, moment_noise=0.01)
    ]

    # Create the dataset.
    dataset = OCNMoleculeDataset(
        root=args.root,
        filenames=train_filenames,
        dataset_size=args.num_samples,
        threshold=2.0,
        num_atoms_in_sample=args.num_atoms_in_sample,
        augmentations=augmentations
    )

    # Split dataset into train and validation
    train_ratio = 0.8
    train_size = int(train_ratio * len(dataset))
    validate_size = len(dataset) - train_size
    train_dataset, validate_dataset = random_split(dataset, [train_size, validate_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=False)

    # Define model
    model = OCNTransformer(n_heads=16, num_layers=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(sum(p.numel() for p in model.parameters()))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(model)

    train_losses = []
    train_r2s = []
    validate_losses = []
    validate_r2s = []

    # Training loop
    for epoch in tqdm(range(1, args.epochs + 1), desc='Training epochs'):
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

        # evaluation
        model.eval()
        epoch_validate_loss = 0.0
        epoch_validate_r2 = 0.0
        total_validate = 0

        validate_preds = []
        validate_targets = []

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
                epoch_validate_loss += loss.item() * batch_size
                total_validate += batch_size

                validate_preds.append(outputs.cpu().numpy())
                validate_targets.append(targets.cpu().numpy())
        
        # epoch validate loss
        epoch_validate_loss /= total_validate
        
        # epoch validate r2
        validate_preds = np.concatenate(validate_preds, axis=0).flatten()
        validate_targets = np.concatenate(validate_targets, axis=0).flatten()
        epoch_validate_r2 = r2(validate_targets, validate_preds)

        validate_losses.append(epoch_validate_loss)
        validate_r2s.append(epoch_validate_r2)

        # Every 10 epochs (and at the final epoch), print progress using tqdm.write
        if epoch % 10 == 0 or epoch == args.epochs:
            tqdm.write(
                f"Epoch {epoch}/{args.epochs}: "
                f"Train Loss: {epoch_train_loss:.4f}, Train R2: {epoch_train_r2:.4f}, "
                f"Validate Loss: {epoch_validate_loss:.4f}, Validate R2: {epoch_validate_r2:.4f}"
            )

    # Save metric to a file
    np.savez('metrics/metrics.npz',
             train_losses=train_losses,
             train_r2s=train_r2s,
             validate_losses=validate_losses,
             validate_r2s=validate_r2s)
    print('Saved training metrics to metrics.npz.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_filenames', nargs='+', help='List of csv files to load for training.')
    parser.add_argument('test_filenames', nargs='+', help='List of csv files to load for testing.')
    parser.add_argument('-r', '--root', default='./raw', help='Root folder for the data.')
    parser.add_argument('-n', '--num-samples', type=int, default=1000, help='Number of samples to generate.')
    parser.add_argument('-a', '--num-atoms-in-sample', type=int, default=16, help='Number of atoms in a sample.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    args = parser.parse_args()

    main(args)