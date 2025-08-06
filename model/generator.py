"""
Molecular Generator for CNO Compounds
Generates new molecular structures using learned patterns from existing molecules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass

from .data_utils import BondCalculator


@dataclass
class GenerationConfig:
    """Configuration for molecular generation."""
    max_atoms: int = 50
    min_atoms: int = 10
    atom_types: List[str] = None
    bond_threshold: float = 2.0
    spatial_scale: float = 10.0
    magnetic_moment_scale: float = 100.0
    
    def __post_init__(self):
        if self.atom_types is None:
            self.atom_types = ['C', 'N', 'O', 'H']


class AtomEncoder:
    """Encode and decode atom types."""
    
    def __init__(self, atom_types: List[str] = None):
        if atom_types is None:
            atom_types = ['C', 'N', 'O', 'H']
        self.atom_types = atom_types
        self.atom_to_idx = {atom: i for i, atom in enumerate(atom_types)}
        self.idx_to_atom = {i: atom for i, atom in enumerate(atom_types)}
        self.num_atom_types = len(atom_types)
    
    def encode(self, atoms: List[str]) -> torch.Tensor:
        """Convert atom types to one-hot encoding."""
        indices = [self.atom_to_idx[atom] for atom in atoms]
        one_hot = torch.zeros(len(atoms), self.num_atom_types)
        one_hot[range(len(atoms)), indices] = 1
        return one_hot
    
    def decode(self, one_hot: torch.Tensor) -> List[str]:
        """Convert one-hot encoding back to atom types."""
        indices = torch.argmax(one_hot, dim=-1)
        return [self.idx_to_atom[idx.item()] for idx in indices]


class MolecularVAE(nn.Module):
    """
    Variational Autoencoder for molecular structures.
    Learns to encode molecular features and generate new molecules.
    """
    
    def __init__(self, config: GenerationConfig):
        super().__init__()
        self.config = config
        self.atom_encoder = AtomEncoder(config.atom_types)
        
        # Input dimensions: atom_type (4) + coordinates (3) + magnetic_moment (1) = 8
        self.input_dim = len(config.atom_types) + 3 + 1
        self.hidden_dim = 128
        self.latent_dim = 32
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        
        # Latent space
        self.mu_layer = nn.Linear(self.hidden_dim, self.latent_dim)
        self.logvar_layer = nn.Linear(self.hidden_dim, self.latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim)
        )
        
        # Output activation for different components
        self.atom_type_activation = nn.Softmax(dim=-1)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space parameters."""
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to molecular features."""
        decoded = self.decoder(z)
        
        # Split output into components
        atom_types = decoded[..., :len(self.config.atom_types)]
        coordinates = decoded[..., len(self.config.atom_types):len(self.config.atom_types)+3]
        magnetic_moment = decoded[..., -1:]
        
        # Apply appropriate activations
        atom_types = self.atom_type_activation(atom_types)
        coordinates = torch.tanh(coordinates) * self.config.spatial_scale
        magnetic_moment = torch.tanh(magnetic_moment) * self.config.magnetic_moment_scale
        
        return torch.cat([atom_types, coordinates, magnetic_moment], dim=-1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def generate(self, num_atoms: int, device: str = 'cpu') -> torch.Tensor:
        """Generate a new molecule by sampling from latent space."""
        z = torch.randn(num_atoms, self.latent_dim, device=device)
        with torch.no_grad():
            generated = self.decode(z)
        return generated


class LocalStructureGenerator:
    """
    Generate local molecular structures around a center atom.
    Uses chemical bonding rules and learned patterns.
    """
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.atom_encoder = AtomEncoder(config.atom_types)
        self.bond_calculator = BondCalculator(config.bond_threshold)
        
        # Chemical bonding preferences
        self.bonding_preferences = {
            'C': {'preferred_neighbors': ['C', 'N', 'O', 'H'], 'max_bonds': 4},
            'N': {'preferred_neighbors': ['C', 'H'], 'max_bonds': 3},
            'O': {'preferred_neighbors': ['C', 'H'], 'max_bonds': 2},
            'H': {'preferred_neighbors': ['C', 'N', 'O'], 'max_bonds': 1}
        }
    
    def generate_local_structure(self, center_atom: str, num_neighbors: int = None, 
                               radius: float = 3.0) -> Dict:
        """
        Generate a local molecular structure around a center atom.
        
        Args:
            center_atom: Type of center atom ('C', 'N', 'O', 'H')
            num_neighbors: Number of neighboring atoms to generate
            radius: Spatial radius for neighbor placement
            
        Returns:
            Dictionary with atoms, coordinates, and estimated magnetic moments
        """
        if center_atom not in self.bonding_preferences:
            raise ValueError(f"Unsupported center atom type: {center_atom}")
        
        prefs = self.bonding_preferences[center_atom]
        
        if num_neighbors is None:
            num_neighbors = random.randint(1, prefs['max_bonds'])
        
        # Generate center atom at origin
        atoms = [center_atom]
        coordinates = [np.array([0.0, 0.0, 0.0])]
        
        # Generate neighbors
        for i in range(num_neighbors):
            # Choose neighbor type based on preferences
            neighbor_type = random.choice(prefs['preferred_neighbors'])
            atoms.append(neighbor_type)
            
            # Generate position around center
            # Use spherical coordinates for realistic positioning
            theta = random.uniform(0, 2 * np.pi)
            phi = random.uniform(0, np.pi)
            r = random.uniform(1.0, radius)
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
            coordinates.append(np.array([x, y, z]))
        
        coordinates = np.array(coordinates)
        
        # Estimate magnetic moments based on atom types and local environment
        magnetic_moments = self._estimate_magnetic_moments(atoms, coordinates)
        
        return {
            'atoms': atoms,
            'coordinates': coordinates,
            'magnetic_moments': magnetic_moments,
            'center_atom': center_atom,
            'num_atoms': len(atoms)
        }
    
    def _estimate_magnetic_moments(self, atoms: List[str], coordinates: np.ndarray) -> np.ndarray:
        """
        Estimate magnetic moments based on atom types and local environment.
        This is a simplified heuristic - in practice, this would be learned.
        """
        magnetic_moments = np.zeros(len(atoms))
        
        # Calculate bonds
        adjacency = self.bond_calculator.calculate_bonds(atoms, coordinates)
        
        for i, atom_type in enumerate(atoms):
            # Base magnetic moment based on atom type
            base_moment = {
                'C': 0.0,  # Carbon typically non-magnetic
                'N': 1.0,  # Nitrogen can be magnetic
                'O': 0.5,  # Oxygen weakly magnetic
                'H': 0.0   # Hydrogen non-magnetic
            }.get(atom_type, 0.0)
            
            # Modify based on local environment
            num_bonds = adjacency[i].sum()
            coordination_factor = 1.0 - (num_bonds / 4.0)  # Less magnetic with more bonds
            
            # Add some randomness
            noise = random.gauss(0, 0.1)
            
            magnetic_moments[i] = base_moment * coordination_factor + noise
        
        return magnetic_moments
    
    def evaluate_structure_quality(self, structure: Dict, reference_molecules: List[Dict]) -> float:
        """
        Evaluate the quality of a generated structure by comparing with reference molecules.
        
        Args:
            structure: Generated molecular structure
            reference_molecules: List of reference molecular structures
            
        Returns:
            Quality score (higher is better)
        """
        if not reference_molecules:
            return 0.0
        
        scores = []
        
        for ref_mol in reference_molecules:
            score = self._compare_structures(structure, ref_mol)
            scores.append(score)
        
        return max(scores)  # Best match score
    
    def _compare_structures(self, struct1: Dict, struct2: Dict) -> float:
        """Compare two molecular structures and return similarity score."""
        # Compare atom type distributions
        atoms1 = struct1['atoms']
        atoms2 = struct2['atoms']
        
        # Count atom types
        count1 = {atom_type: atoms1.count(atom_type) for atom_type in self.config.atom_types}
        count2 = {atom_type: atoms2.count(atom_type) for atom_type in self.config.atom_types}
        
        # Calculate composition similarity
        total1 = sum(count1.values())
        total2 = sum(count2.values())
        
        if total1 == 0 or total2 == 0:
            return 0.0
        
        composition_score = 0.0
        for atom_type in self.config.atom_types:
            frac1 = count1.get(atom_type, 0) / total1
            frac2 = count2.get(atom_type, 0) / total2
            composition_score += 1.0 - abs(frac1 - frac2)
        
        composition_score /= len(self.config.atom_types)
        
        # Compare bonding patterns
        bond_score = self._compare_bonding_patterns(struct1, struct2)
        
        # Combine scores
        return 0.7 * composition_score + 0.3 * bond_score
    
    def _compare_bonding_patterns(self, struct1: Dict, struct2: Dict) -> float:
        """Compare bonding patterns between two structures."""
        adj1 = self.bond_calculator.calculate_bonds(struct1['atoms'], struct1['coordinates'])
        adj2 = self.bond_calculator.calculate_bonds(struct2['atoms'], struct2['coordinates'])
        
        # Calculate average coordination numbers for each atom type
        coord1 = self._get_coordination_stats(struct1['atoms'], adj1)
        coord2 = self._get_coordination_stats(struct2['atoms'], adj2)
        
        score = 0.0
        count = 0
        
        for atom_type in self.config.atom_types:
            if atom_type in coord1 and atom_type in coord2:
                diff = abs(coord1[atom_type] - coord2[atom_type])
                score += 1.0 - min(diff / 4.0, 1.0)  # Normalize by max coordination
                count += 1
        
        return score / count if count > 0 else 0.0
    
    def _get_coordination_stats(self, atoms: List[str], adjacency: np.ndarray) -> Dict[str, float]:
        """Get average coordination numbers for each atom type."""
        coords = {}
        
        for atom_type in set(atoms):
            indices = [i for i, a in enumerate(atoms) if a == atom_type]
            if indices:
                avg_coord = np.mean([adjacency[i].sum() for i in indices])
                coords[atom_type] = avg_coord
        
        return coords


if __name__ == "__main__":
    # Test the generators
    config = GenerationConfig(max_atoms=20, min_atoms=5)
    
    # Test local structure generator
    local_gen = LocalStructureGenerator(config)
    
    print("Testing Local Structure Generator:")
    for center_atom in ['C', 'N', 'O']:
        structure = local_gen.generate_local_structure(center_atom, num_neighbors=3)
        print(f"\nGenerated structure around {center_atom}:")
        print(f"Atoms: {structure['atoms']}")
        print(f"Coordinates shape: {structure['coordinates'].shape}")
        print(f"Magnetic moments: {structure['magnetic_moments']}")
    
    # Test VAE
    print("\nTesting Molecular VAE:")
    vae = MolecularVAE(config)
    
    # Create dummy input
    dummy_input = torch.randn(10, 8)  # 10 atoms, 8 features each
    recon, mu, logvar = vae(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent mu shape: {mu.shape}")
    
    # Test generation
    generated = vae.generate(num_atoms=15)
    print(f"Generated molecule shape: {generated.shape}")
