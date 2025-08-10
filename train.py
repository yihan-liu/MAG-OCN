# train.py

import os
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_util.preprocessor import OCNMoleculeDataset, OCNSpatialSegmentDataset
from model.chemberta_ft_model import MAGChemBERTa
from utils import mol_to_explicit_smiles, token2atom_mapping, collate

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Add .csv extension to filenames if not present
    csv_filenames = []
    for fname in args.filenames:
        if not fname.endswith('.csv'):
            csv_filenames.append(fname + '.csv')
        else:
            csv_filenames.append(fname)
    
    # Load dataset with spatial segmentation
    ds = OCNSpatialSegmentDataset(
        root=args.root,
        filenames=csv_filenames,
        processed_dir=args.processed_dir,
        augmentations=None,
        max_atoms_per_segment=args.max_atoms_per_segment,
    )

    tokenizer = MAGChemBERTa.get_tokenizer(args.pretrained)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate(x, tokenizer)
    )

    model = MAGChemBERTa(pretrained_name=args.pretrained, lora_r=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_mm = 0
        n_tokens = 0

        for batch in dl:
            # Move tensors to device, but keep token2atom as list
            token2atom = batch.pop('token2atom')  # Remove from batch before device transfer
            batch = {k: v.to(device) for k, v in batch.items()}
            
            mm_pred = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                coords=batch['coords'],
                token2atom=token2atom,  # Pass as list, not moved to device
                mask=batch['mask'],
            )

            mask = batch['mask'].bool()  # Convert to boolean for proper indexing
            loss = F.mse_loss(mm_pred[mask], batch['mm_reduced'][mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_mm += loss.item() * mask.sum().item()
            n_tokens += mask.sum().item()

        print(f"Epoch {epoch:02d} | atom-MSE {total_mm/n_tokens:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ChemBERTa with OCN data")
    parser.add_argument('--root', default='./raw')
    parser.add_argument('--filenames', nargs='+', required=True, help='List of CSV files with molecule data (extension optional)')
    parser.add_argument('--processed-dir', default='./processed', help='Directory to save processed data')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--pretrained', default='seyonec/ChemBERTa-zinc-base-v1', help='Pretrained ChemBERTa model name')
    parser.add_argument('--max-atoms-per-segment', type=int, default=100, 
                        help='Maximum atoms per spatial segment (affects max token length)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    train(args)