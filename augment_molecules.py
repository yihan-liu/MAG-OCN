"""
Main augmentation pipeline script
Loads molecular data, trains generative models, and creates augmented datasets.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from typing import Dict, List
import pickle

# Add model directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.data_utils import MolecularData, BondCalculator
from model.generator import GenerationConfig, LocalStructureGenerator
from model.augmentation import MolecularAugmentationSystem, AugmentationConfig


def save_augmented_dataset(molecules: List[Dict], output_path: str):
    """Save augmented molecules to CSV files."""
    os.makedirs(output_path, exist_ok=True)
    
    for i, mol in enumerate(molecules):
        # Create DataFrame
        data = []
        for j, (atom, coord, moment) in enumerate(zip(
            mol['atoms'], mol['coordinates'], mol['magnetic_moments']
        )):
            data.append({
                'ATOM': f"{atom}{j+1}",
                'X': coord[0],
                'Y': coord[1], 
                'Z': coord[2],
                'MAGNETIC_MOMENT': moment
            })
        
        df = pd.DataFrame(data)
        
        # Determine filename
        if mol.get('generated', False):
            filename = f"generated_{i}.csv"
        elif mol.get('mixed', False):
            filename = f"mixed_{i}.csv"
        else:
            v_value = mol.get('v_value', 0)
            filename = f"{v_value}v_augmented_{i}.csv"
        
        filepath = os.path.join(output_path, filename)
        df.to_csv(filepath, index=False)
    
    print(f"Saved {len(molecules)} augmented molecules to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Molecular Augmentation Pipeline")
    parser.add_argument("--data_dir", default="raw", 
                       help="Directory containing molecular CSV files")
    parser.add_argument("--output_dir", default="augmented_data",
                       help="Directory to save augmented molecules")
    parser.add_argument("--target_size", type=int, default=100,
                       help="Target number of molecules in augmented dataset")
    parser.add_argument("--train_vae", action="store_true",
                       help="Train VAE on reference molecules")
    parser.add_argument("--vae_epochs", type=int, default=50,
                       help="Number of epochs to train VAE")
    parser.add_argument("--save_models", action="store_true",
                       help="Save trained models")
    parser.add_argument("--model_dir", default="trained_models",
                       help="Directory to save/load models")
    parser.add_argument("--use_expanded_only", action="store_true", default=True,
                       help="Use only datasets ending with '_expanded' (default: True)")
    parser.add_argument("--use_all_datasets", action="store_true",
                       help="Use all datasets, not just expanded ones")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("Molecular Augmentation Pipeline")
    print("=" * 40)
    
    # Load reference molecules
    print("Loading reference molecules...")
    data_loader = MolecularData(args.data_dir)
    
    # Choose loading method based on arguments
    if args.use_all_datasets:
        molecules_dict = data_loader.load_all_molecules()
        print("Using all datasets")
    else:
        molecules_dict = data_loader.load_expanded_molecules()
        print("Using only expanded datasets")
        
    molecules_list = list(molecules_dict.values())
    
    print(f"Loaded {len(molecules_list)} reference molecules")
    
    # Display dataset statistics
    stats = data_loader.get_molecule_stats()
    print(f"\nDataset Statistics:")
    print(f"  Atom counts: {stats['atom_counts'].min()}-{stats['atom_counts'].max()} "
          f"(mean: {stats['atom_counts'].mean():.1f})")
    print(f"  V-values: {stats['v_values'].min()}-{stats['v_values'].max()}")
    print(f"  Atom types: {dict(stats['atom_type_distribution'])}")
    print(f"  Magnetic moment range: {stats['magnetic_moment_stats'].min():.2f} to "
          f"{stats['magnetic_moment_stats'].max():.2f}")
    
    # Setup augmentation system
    print(f"\nSetting up augmentation system...")
    generation_config = GenerationConfig(
        max_atoms=50,
        min_atoms=5,
        atom_types=['C', 'N', 'O', 'H'],
        bond_threshold=2.0,
        spatial_scale=10.0,
        magnetic_moment_scale=100.0
    )
    
    augmentation_config = AugmentationConfig(
        rotation_prob=0.7,
        translation_prob=0.5,
        noise_prob=0.6,
        position_noise_std=0.1,
        moment_noise_std=0.2,
        generation_prob=0.3,
        local_generation_prob=0.4,
        quality_threshold=0.4,
        mixing_prob=0.2
    )
    
    aug_system = MolecularAugmentationSystem(
        reference_molecules=molecules_list,
        generation_config=generation_config,
        augmentation_config=augmentation_config
    )
    
    # Train VAE if requested
    if args.train_vae:
        print(f"\nTraining VAE for {args.vae_epochs} epochs...")
        aug_system.train_vae(
            molecules=molecules_list,
            epochs=args.vae_epochs,
            batch_size=16,
            learning_rate=1e-3
        )
        
        if args.save_models:
            os.makedirs(args.model_dir, exist_ok=True)
            model_path = os.path.join(args.model_dir, "molecular_vae.pth")
            torch.save(aug_system.generative_aug.vae.state_dict(), model_path)
            print(f"Saved VAE model to {model_path}")
    
    # Generate augmented dataset
    print(f"\nGenerating augmented dataset (target size: {args.target_size})...")
    augmented_molecules = aug_system.augment_dataset(
        molecules=molecules_list,
        target_size=args.target_size
    )
    
    print(f"Generated {len(augmented_molecules)} total molecules")
    
    # Analyze augmented dataset
    print(f"\nAugmented Dataset Analysis:")
    original_count = len(molecules_list)
    generated_count = len([m for m in augmented_molecules if m.get('generated', False)])
    mixed_count = len([m for m in augmented_molecules if m.get('mixed', False)])
    traditional_count = len(augmented_molecules) - original_count - generated_count - mixed_count
    
    print(f"  Original molecules: {original_count}")
    print(f"  Traditional augmentations: {traditional_count}")
    print(f"  Generated molecules: {generated_count}")
    print(f"  Mixed molecules: {mixed_count}")
    
    # Calculate diversity metrics
    all_atom_types = set()
    total_atoms = 0
    magnetic_moments = []
    
    for mol in augmented_molecules:
        all_atom_types.update(mol['atoms'])
        total_atoms += mol['num_atoms']
        magnetic_moments.extend(mol['magnetic_moments'])
    
    magnetic_moments = np.array(magnetic_moments)
    
    print(f"  Total atoms in dataset: {total_atoms}")
    print(f"  Unique atom types: {sorted(all_atom_types)}")
    print(f"  Magnetic moment stats: min={magnetic_moments.min():.2f}, "
          f"max={magnetic_moments.max():.2f}, std={magnetic_moments.std():.2f}")
    
    # Save augmented dataset
    print(f"\nSaving augmented dataset...")
    save_augmented_dataset(augmented_molecules, args.output_dir)
    
    # Save metadata
    metadata = {
        'original_count': original_count,
        'augmented_count': len(augmented_molecules),
        'generation_config': generation_config,
        'augmentation_config': augmentation_config,
        'dataset_stats': stats
    }
    
    metadata_path = os.path.join(args.output_dir, "metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Saved metadata to {metadata_path}")
    print(f"\nAugmentation pipeline completed successfully!")


if __name__ == "__main__":
    main()
