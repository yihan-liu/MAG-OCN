#!/usr/bin/env python3
"""
Test script for molecular augmentation system
"""

import sys
import os
import pandas as pd

# Add parent directory to path to import model modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.data_utils import MolecularData
from model.augmentation import MolecularAugmentationSystem, AugmentationConfig
from model.generator import GenerationConfig

def test_basic_functionality():
    """Test basic functionality of the augmentation system"""
    print("Testing molecular augmentation system...")
    
    # Load a sample molecule
    try:
        # Get the project root directory (parent of test directory)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw_data_path = os.path.join(project_root, "raw")
        
        loader = MolecularData(raw_data_path)
        molecules_dict = loader.load_expanded_molecules()
        
        if not molecules_dict:
            raise ValueError("No '_expanded' datasets found in the raw directory")
        
        print(f"Found {len(molecules_dict)} expanded datasets: {list(molecules_dict.keys())}")
        
        # Get a sample molecule (use the first expanded one)
        sample_key = list(molecules_dict.keys())[0]
        molecule = molecules_dict[sample_key]
        print(f"✓ Loaded molecule '{sample_key}' with {molecule['num_atoms']} atoms")
        
        # Setup augmentation system
        generation_config = GenerationConfig(
            max_atoms=50,
            min_atoms=5,
            atom_types=['C', 'N', 'O', 'H'],
            bond_threshold=2.0
        )
        
        augmentation_config = AugmentationConfig(
            # Set all probabilities to 1.0 to ensure all methods are tested
            rotation_prob=1.0,
            translation_prob=1.0,
            noise_prob=1.0,
            generation_prob=1.0,
            local_generation_prob=1.0,
            mixing_prob=1.0
        )
        
        # Create augmentation system with all molecules as reference
        augmenter = MolecularAugmentationSystem(
            reference_molecules=list(molecules_dict.values()),
            generation_config=generation_config,
            augmentation_config=augmentation_config
        )
        
        # Train VAE before using it for generation
        print("Training VAE on reference molecules...")
        augmenter.train_vae(list(molecules_dict.values()), epochs=20, batch_size=4)
        print("✓ VAE training completed")
        
        # Test all augmentation methods on a single molecule
        print(f"\nTesting all augmentation methods on '{sample_key}'...")
        augmented = augmenter.augment_molecule(molecule)
        print(f"✓ Generated {len(augmented)} augmented molecules using all methods")
        
        # Display some statistics
        original_atoms = molecule['atoms']
        original_c_count = original_atoms.count('C')
        original_n_count = original_atoms.count('N')
        original_o_count = original_atoms.count('O')
        original_h_count = original_atoms.count('H')
        
        print(f"\nOriginal molecule '{sample_key}' composition:")
        print(f"  - Carbon (C): {original_c_count}")
        print(f"  - Nitrogen (N): {original_n_count}")
        print(f"  - Oxygen (O): {original_o_count}")
        print(f"  - Hydrogen (H): {original_h_count}")
        print(f"  - Magnetic moment range: {min(molecule['magnetic_moments']):.2f} to {max(molecule['magnetic_moments']):.2f}")
        
        print(f"\nAugmented molecules analysis:")
        for i, aug_mol in enumerate(augmented):
            aug_atoms = aug_mol['atoms']
            aug_c_count = aug_atoms.count('C')
            aug_n_count = aug_atoms.count('N')
            aug_o_count = aug_atoms.count('O')
            aug_h_count = aug_atoms.count('H')
            
            mol_type = "Traditional Augmentation"
            if aug_mol.get('generated', False):
                mol_type = "VAE Generated"
            elif aug_mol.get('mixed', False):
                mol_type = "Molecule Mixing"
            elif hasattr(aug_mol, '_is_local_generated'):
                mol_type = "Local Structure Generated"
            
            print(f"Molecule {i+1} ({mol_type}) - {aug_mol['num_atoms']} atoms:")
            print(f"  - C: {aug_c_count}, N: {aug_n_count}, O: {aug_o_count}, H: {aug_h_count}")
            print(f"  - Magnetic moment range: {min(aug_mol['magnetic_moments']):.2f} to {max(aug_mol['magnetic_moments']):.2f}")
        
        # Test single molecule dataset augmentation
        print(f"\nTesting single molecule dataset augmentation...")
        single_mol_augmented = augmenter.augment_single_molecule_dataset(molecule, target_count=5)
        print(f"✓ Generated {len(single_mol_augmented)} molecules from single input molecule")
        
        print("\n✓ All augmentation methods tested successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_functionality()
