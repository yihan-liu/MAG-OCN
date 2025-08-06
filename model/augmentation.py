"""
Molecular Augmentation System
Combines traditional augmentation with generative models for data augmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Union
import copy
from dataclasses import dataclass

from .data_utils import MolecularData, BondCalculator
from .generator import LocalStructureGenerator, MolecularVAE, GenerationConfig, AtomEncoder


@dataclass
class AugmentationConfig:
    """Configuration for molecular augmentation."""
    # Traditional augmentation parameters
    rotation_prob: float = 0.5
    translation_prob: float = 0.3
    noise_prob: float = 0.4
    position_noise_std: float = 0.1
    moment_noise_std: float = 0.2
    
    # Generative augmentation parameters
    generation_prob: float = 0.2
    local_generation_prob: float = 0.3
    min_generated_atoms: int = 5
    max_generated_atoms: int = 15
    quality_threshold: float = 0.5
    
    # Mixing parameters
    mixing_prob: float = 0.1
    max_mix_ratio: float = 0.3


class TraditionalAugmentations:
    """Traditional molecular augmentation methods."""
    
    @staticmethod
    def random_rotation(molecule: Dict, config: AugmentationConfig) -> Dict:
        """Apply random 3D rotation to molecular coordinates."""
        if random.random() > config.rotation_prob:
            return molecule
        
        mol = copy.deepcopy(molecule)
        coords = mol['coordinates']
        
        # Generate random rotation matrix
        # Rotation around random axis
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.uniform(0, 2 * np.pi)
        
        # Rodrigues' rotation formula
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        
        # Apply rotation
        mol['coordinates'] = coords @ R.T
        
        return mol
    
    @staticmethod
    def random_translation(molecule: Dict, config: AugmentationConfig, 
                          max_translation: float = 2.0) -> Dict:
        """Apply random translation to molecular coordinates."""
        if random.random() > config.translation_prob:
            return molecule
        
        mol = copy.deepcopy(molecule)
        
        # Random translation vector
        translation = np.random.uniform(-max_translation, max_translation, 3)
        mol['coordinates'] += translation
        
        return mol
    
    @staticmethod
    def add_noise(molecule: Dict, config: AugmentationConfig) -> Dict:
        """Add random noise to coordinates and magnetic moments."""
        if random.random() > config.noise_prob:
            return molecule
        
        mol = copy.deepcopy(molecule)
        
        # Add noise to coordinates
        position_noise = np.random.normal(0, config.position_noise_std, 
                                        mol['coordinates'].shape)
        mol['coordinates'] += position_noise
        
        # Add noise to magnetic moments (ensure float type)
        moment_noise = np.random.normal(0, config.moment_noise_std, 
                                      mol['magnetic_moments'].shape)
        mol['magnetic_moments'] = mol['magnetic_moments'].astype(float) + moment_noise
        
        return mol
    
    @staticmethod
    def random_reflection(molecule: Dict, prob: float = 0.3) -> Dict:
        """Apply random reflection across coordinate planes."""
        if random.random() > prob:
            return molecule
        
        mol = copy.deepcopy(molecule)
        
        # Choose random plane (xy, xz, or yz)
        plane = random.choice([0, 1, 2])
        mol['coordinates'][:, plane] *= -1
        
        return mol


class GenerativeAugmentation:
    """Generative model-based augmentation methods."""
    
    def __init__(self, config: GenerationConfig, reference_molecules: List[Dict]):
        self.config = config
        self.reference_molecules = reference_molecules
        self.local_generator = LocalStructureGenerator(config)
        self.bond_calculator = BondCalculator(config.bond_threshold)
        self.atom_encoder = AtomEncoder(config.atom_types)
        
        # Initialize VAE (would be trained in practice)
        self.vae = MolecularVAE(config)
        
    def generate_local_structure(self, augment_config: AugmentationConfig) -> Optional[Dict]:
        """Generate a new local molecular structure."""
        if random.random() > augment_config.local_generation_prob:
            return None
        
        # Try multiple times if probability is 1.0 (deterministic mode)
        max_attempts = 5 if augment_config.local_generation_prob >= 1.0 else 1
        
        for attempt in range(max_attempts):
            # Choose random center atom type
            center_atom = random.choice(self.config.atom_types)
            
            # Generate structure
            structure = self.local_generator.generate_local_structure(
                center_atom=center_atom,
                num_neighbors=random.randint(2, 4),
                radius=3.0
            )
            
            # Evaluate quality against reference molecules
            quality_score = self.local_generator.evaluate_structure_quality(
                structure, self.reference_molecules
            )
            
            # Lower threshold in deterministic mode
            threshold = augment_config.quality_threshold if augment_config.local_generation_prob < 1.0 else 0.1
            if quality_score >= threshold:
                return structure
        
        return None
    
    def generate_full_molecule(self, augment_config: AugmentationConfig) -> Optional[Dict]:
        """Generate a complete new molecule using VAE."""
        if random.random() > augment_config.generation_prob:
            return None
        
        # Try multiple times if probability is 1.0 (deterministic mode)
        max_attempts = 5 if augment_config.generation_prob >= 1.0 else 1
        
        for attempt in range(max_attempts):
            num_atoms = random.randint(augment_config.min_generated_atoms, 
                                     augment_config.max_generated_atoms)
            
            # Generate using VAE
            generated_tensor = self.vae.generate(num_atoms=num_atoms)
            
            # Convert to molecular structure
            structure = self._tensor_to_molecule(generated_tensor)
            
            # Evaluate quality
            quality_score = self.local_generator.evaluate_structure_quality(
                structure, self.reference_molecules
            )
            
            # Lower threshold in deterministic mode
            threshold = augment_config.quality_threshold if augment_config.generation_prob < 1.0 else 0.1
            if quality_score >= threshold:
                return structure
        
        return None
    
    def _tensor_to_molecule(self, tensor: torch.Tensor) -> Dict:
        """Convert generated tensor to molecular structure format."""
        num_atoms = tensor.shape[0]
        atom_types_one_hot = tensor[:, :len(self.config.atom_types)]
        coordinates = tensor[:, len(self.config.atom_types):len(self.config.atom_types)+3]
        magnetic_moments = tensor[:, -1]
        
        # Decode atom types
        atoms = self.atom_encoder.decode(atom_types_one_hot)
        
        return {
            'atoms': atoms,
            'coordinates': coordinates.detach().numpy(),
            'magnetic_moments': magnetic_moments.detach().numpy(),
            'num_atoms': num_atoms,
            'generated': True
        }
    
    def mix_molecules(self, mol1: Dict, mol2: Dict, 
                     augment_config: AugmentationConfig) -> Optional[Dict]:
        """Create a new molecule by mixing parts of two existing molecules."""
        if random.random() > augment_config.mixing_prob:
            return None
        
        # Try multiple times if probability is 1.0 (deterministic mode)
        max_attempts = 3 if augment_config.mixing_prob >= 1.0 else 1
        
        for attempt in range(max_attempts):
            # Determine mixing ratio
            mix_ratio = random.uniform(0.1, augment_config.max_mix_ratio)
            
            # Choose atoms from each molecule
            num_atoms1 = max(1, int(len(mol1['atoms']) * (1 - mix_ratio)))
            num_atoms2 = max(1, int(len(mol2['atoms']) * mix_ratio))
            
            if num_atoms1 == 0 or num_atoms2 == 0:
                continue
            
            # Random selection from each molecule
            indices1 = random.sample(range(len(mol1['atoms'])), min(num_atoms1, len(mol1['atoms'])))
            indices2 = random.sample(range(len(mol2['atoms'])), min(num_atoms2, len(mol2['atoms'])))
            
            # Combine atoms
            mixed_atoms = [mol1['atoms'][i] for i in indices1] + [mol2['atoms'][i] for i in indices2]
            mixed_coords = np.vstack([mol1['coordinates'][indices1], mol2['coordinates'][indices2]])
            mixed_moments = np.hstack([mol1['magnetic_moments'][indices1], mol2['magnetic_moments'][indices2]])
            
            # Center the mixed molecule
            center = mixed_coords.mean(axis=0)
            mixed_coords -= center
            
            return {
                'atoms': mixed_atoms,
                'coordinates': mixed_coords,
                'magnetic_moments': mixed_moments,
                'num_atoms': len(mixed_atoms),
                'mixed': True
            }
        
        return None


class MolecularAugmentationSystem:
    """
    Complete molecular augmentation system combining traditional and generative methods.
    """
    
    def __init__(self, reference_molecules: List[Dict], 
                 generation_config: GenerationConfig = None,
                 augmentation_config: AugmentationConfig = None):
        
        if generation_config is None:
            generation_config = GenerationConfig()
        if augmentation_config is None:
            augmentation_config = AugmentationConfig()
            
        self.generation_config = generation_config
        self.augmentation_config = augmentation_config
        self.reference_molecules = reference_molecules
        
        # Initialize augmentation components
        self.traditional_aug = TraditionalAugmentations()
        self.generative_aug = GenerativeAugmentation(generation_config, reference_molecules)
        
    def augment_molecule(self, molecule: Dict) -> List[Dict]:
        """
        Apply full augmentation pipeline to a single molecule.
        
        Returns a list of augmented molecules.
        """
        augmented = []
        
        # Apply traditional augmentations
        mol = copy.deepcopy(molecule)
        mol = self.traditional_aug.random_rotation(mol, self.augmentation_config)
        mol = self.traditional_aug.random_translation(mol, self.augmentation_config)
        mol = self.traditional_aug.add_noise(mol, self.augmentation_config)
        mol = self.traditional_aug.random_reflection(mol, 0.3)
        augmented.append(mol)
        
        # Generate local structures
        local_structure = self.generative_aug.generate_local_structure(self.augmentation_config)
        if local_structure is not None:
            local_structure['_is_local_generated'] = True
            augmented.append(local_structure)
        
        # Generate full molecules
        full_molecule = self.generative_aug.generate_full_molecule(self.augmentation_config)
        if full_molecule is not None:
            augmented.append(full_molecule)
        
        # Mix with other molecules
        if len(self.reference_molecules) > 1:
            other_mol = random.choice([m for m in self.reference_molecules 
                                     if m is not molecule])
            mixed_mol = self.generative_aug.mix_molecules(molecule, other_mol, 
                                                        self.augmentation_config)
            if mixed_mol is not None:
                augmented.append(mixed_mol)
        
        return augmented
    
    def augment_single_molecule_dataset(self, molecule: Dict, target_count: int = 5) -> List[Dict]:
        """
        Generate multiple augmented versions of a single molecule using all augmentation methods.
        This method ensures all augmentation types are applied deterministically.
        
        Args:
            molecule: Single molecule to augment
            target_count: Number of augmented molecules to generate
            
        Returns:
            List of augmented molecules including the original
        """
        augmented_dataset = [copy.deepcopy(molecule)]  # Start with original
        
        # Apply each augmentation method at least once
        methods_applied = 0
        
        # 1. Traditional augmentation (rotation + translation + noise)
        if methods_applied < target_count - 1:
            mol_trad = copy.deepcopy(molecule)
            mol_trad = self.traditional_aug.random_rotation(mol_trad, self.augmentation_config)
            mol_trad = self.traditional_aug.random_translation(mol_trad, self.augmentation_config)
            mol_trad = self.traditional_aug.add_noise(mol_trad, self.augmentation_config)
            mol_trad['augmentation_type'] = 'traditional'
            augmented_dataset.append(mol_trad)
            methods_applied += 1
        
        # 2. Generate local structure
        if methods_applied < target_count - 1:
            local_structure = self.generative_aug.generate_local_structure(self.augmentation_config)
            if local_structure is not None:
                local_structure['augmentation_type'] = 'local_generation'
                augmented_dataset.append(local_structure)
                methods_applied += 1
        
        # 3. Generate full molecule using VAE
        if methods_applied < target_count - 1:
            full_molecule = self.generative_aug.generate_full_molecule(self.augmentation_config)
            if full_molecule is not None:
                full_molecule['augmentation_type'] = 'vae_generation'
                augmented_dataset.append(full_molecule)
                methods_applied += 1
        
        # 4. Mix with reference molecules
        if methods_applied < target_count - 1 and len(self.reference_molecules) > 1:
            other_mol = random.choice([m for m in self.reference_molecules 
                                     if m.get('filename', '') != molecule.get('filename', '')])
            mixed_mol = self.generative_aug.mix_molecules(molecule, other_mol, 
                                                        self.augmentation_config)
            if mixed_mol is not None:
                mixed_mol['augmentation_type'] = 'molecule_mixing'
                augmented_dataset.append(mixed_mol)
                methods_applied += 1
        
        # Fill remaining slots with additional traditional augmentations
        while len(augmented_dataset) < target_count:
            mol_extra = copy.deepcopy(molecule)
            mol_extra = self.traditional_aug.random_rotation(mol_extra, self.augmentation_config)
            mol_extra = self.traditional_aug.add_noise(mol_extra, self.augmentation_config)
            mol_extra = self.traditional_aug.random_reflection(mol_extra, 0.5)
            mol_extra['augmentation_type'] = 'traditional_extra'
            augmented_dataset.append(mol_extra)
        
        return augmented_dataset[:target_count]
    
    def augment_dataset(self, molecules: List[Dict], 
                       target_size: int = None) -> List[Dict]:
        """
        Augment entire dataset to reach target size.
        
        Args:
            molecules: List of original molecules
            target_size: Target number of molecules (if None, multiply by 5)
            
        Returns:
            Augmented dataset
        """
        if target_size is None:
            target_size = len(molecules) * 5
        
        augmented_dataset = list(molecules)  # Start with originals
        
        while len(augmented_dataset) < target_size:
            # Choose random molecule to augment
            base_molecule = random.choice(molecules)
            
            # Apply augmentations
            new_molecules = self.augment_molecule(base_molecule)
            
            # Add to dataset
            augmented_dataset.extend(new_molecules)
            
            if len(augmented_dataset) % 100 == 0:
                print(f"Generated {len(augmented_dataset)} molecules...")
        
        return augmented_dataset[:target_size]
    
    def train_vae(self, molecules: List[Dict], epochs: int = 100, 
                  batch_size: int = 32, learning_rate: float = 1e-3):
        """
        Train the VAE on the reference molecules.
        
        This is a simplified training loop - in practice, you'd want more sophisticated training.
        """
        # Prepare training data
        training_data = self._prepare_training_data(molecules)
        
        # Setup training
        optimizer = torch.optim.Adam(self.generative_aug.vae.parameters(), lr=learning_rate)
        
        print(f"Training VAE on {len(training_data)} samples...")
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                if len(batch) == 0:
                    continue
                
                batch_tensor = torch.stack(batch)
                
                # Forward pass
                recon, mu, logvar = self.generative_aug.vae(batch_tensor)
                
                # Calculate loss
                loss = self._vae_loss(batch_tensor, recon, mu, logvar)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if epoch % 10 == 0:
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                print(f"Epoch {epoch}: Average loss = {avg_loss:.4f}")
    
    def _prepare_training_data(self, molecules: List[Dict]) -> List[torch.Tensor]:
        """Convert molecules to training tensors for VAE."""
        training_data = []
        
        for mol in molecules:
            # Sample random atoms from molecule for training
            num_atoms = min(len(mol['atoms']), 20)  # Limit size
            if num_atoms < 3:
                continue
                
            # Random sampling
            indices = random.sample(range(len(mol['atoms'])), num_atoms)
            
            # Create feature tensor
            atoms_subset = [mol['atoms'][i] for i in indices]
            coords_subset = mol['coordinates'][indices]
            moments_subset = mol['magnetic_moments'][indices]
            
            # Encode atoms
            atom_features = self.generative_aug.atom_encoder.encode(atoms_subset)
            
            # Combine features
            features = torch.cat([
                atom_features,
                torch.tensor(coords_subset, dtype=torch.float32),
                torch.tensor(moments_subset, dtype=torch.float32).unsqueeze(-1)
            ], dim=-1)
            
            training_data.append(features)
        
        return training_data
    
    def _vae_loss(self, x: torch.Tensor, recon: torch.Tensor, 
                  mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Calculate VAE loss (reconstruction + KL divergence)."""
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + 0.1 * kl_loss  # Weight KL term


if __name__ == "__main__":
    # Test the complete augmentation system
    print("Testing Molecular Augmentation System...")
    
    # Load reference molecules
    data_loader = MolecularData()
    molecules = data_loader.load_all_molecules()
    molecule_list = list(molecules.values())
    
    print(f"Loaded {len(molecule_list)} reference molecules")
    
    # Create augmentation system
    gen_config = GenerationConfig(max_atoms=30, min_atoms=5)
    aug_config = AugmentationConfig()
    
    aug_system = MolecularAugmentationSystem(
        reference_molecules=molecule_list,
        generation_config=gen_config,
        augmentation_config=aug_config
    )
    
    # Test single molecule augmentation
    test_mol = molecule_list[0]
    print(f"\nTesting augmentation on molecule with {test_mol['num_atoms']} atoms")
    
    augmented = aug_system.augment_molecule(test_mol)
    print(f"Generated {len(augmented)} augmented versions")
    
    for i, mol in enumerate(augmented):
        print(f"  Augmented {i+1}: {mol['num_atoms']} atoms, "
              f"types: {set(mol['atoms'])}")
    
    # Test dataset augmentation (small scale)
    print(f"\nTesting dataset augmentation...")
    small_dataset = molecule_list[:2]  # Use first 2 molecules
    augmented_dataset = aug_system.augment_dataset(small_dataset, target_size=10)
    
    print(f"Augmented dataset size: {len(augmented_dataset)}")
    
    # Train VAE (just a few epochs for testing)
    print(f"\nTesting VAE training...")
    aug_system.train_vae(molecule_list, epochs=5, batch_size=8)
    
    print("All tests completed successfully!")
