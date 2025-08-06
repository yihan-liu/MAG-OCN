"""
Molecular Data Utilities for CNO Compounds
Provides functionality to load, process, and analyze molecular structure data.
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import networkx as nx


class MolecularData:
    """
    Class to handle molecular data loading and basic processing.
    """
    
    def __init__(self, data_dir: str = "raw"):
        self.data_dir = data_dir
        self.molecules = {}
        self.atom_types = {'C', 'N', 'O', 'H'}
        
    def load_all_molecules(self) -> Dict[str, Dict]:
        """Load all CSV files from the data directory."""
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        for filename in csv_files:
            molecule_name = filename.replace('.csv', '')
            filepath = os.path.join(self.data_dir, filename)
            self.molecules[molecule_name] = self.load_molecule(filepath)
            
        print(f"Loaded {len(self.molecules)} molecules")
        return self.molecules
    
    def load_expanded_molecules(self) -> Dict[str, Dict]:
        """Load only CSV files that end with '_expanded'."""
        csv_files = [f for f in os.listdir(self.data_dir) 
                    if f.endswith('.csv') and f.replace('.csv', '').endswith('_expanded')]
        
        expanded_molecules = {}
        for filename in csv_files:
            molecule_name = filename.replace('.csv', '')
            filepath = os.path.join(self.data_dir, filename)
            expanded_molecules[molecule_name] = self.load_molecule(filepath)
            
        print(f"Loaded {len(expanded_molecules)} expanded molecules")
        return expanded_molecules
    
    def load_molecule(self, filepath: str) -> Dict:
        """Load a single molecule from CSV file."""
        df = pd.read_csv(filepath)
        
        # Extract v-value from filename (number before 'v')
        filename = os.path.basename(filepath)
        try:
            v_value = int(filename.split('v')[0])
        except:
            v_value = 0
            
        # Parse atom data
        atoms = []
        coordinates = []
        magnetic_moments = []
        
        for _, row in df.iterrows():
            atom_full = row['ATOM']
            atom_type = atom_full[0]  # First character is the element
            
            if atom_type in self.atom_types:
                atoms.append(atom_type)
                coordinates.append([row['X'], row['Y'], row['Z']])
                magnetic_moments.append(row['MAGNETIC_MOMENT'])
        
        coordinates = np.array(coordinates)
        magnetic_moments = np.array(magnetic_moments)
        
        # Center the molecule
        center = coordinates.mean(axis=0)
        coordinates -= center
        
        return {
            'atoms': atoms,
            'coordinates': coordinates,
            'magnetic_moments': magnetic_moments,
            'v_value': v_value,
            'num_atoms': len(atoms),
            'filename': filename
        }
    
    def get_molecule_stats(self) -> Dict:
        """Get statistics about the loaded molecules."""
        if not self.molecules:
            self.load_all_molecules()
            
        stats = {
            'total_molecules': len(self.molecules),
            'atom_counts': [],
            'v_values': [],
            'atom_type_distribution': defaultdict(int),
            'magnetic_moment_stats': []
        }
        
        for mol_data in self.molecules.values():
            stats['atom_counts'].append(mol_data['num_atoms'])
            stats['v_values'].append(mol_data['v_value'])
            stats['magnetic_moment_stats'].extend(mol_data['magnetic_moments'])
            
            for atom_type in mol_data['atoms']:
                stats['atom_type_distribution'][atom_type] += 1
                
        stats['atom_counts'] = np.array(stats['atom_counts'])
        stats['v_values'] = np.array(stats['v_values'])
        stats['magnetic_moment_stats'] = np.array(stats['magnetic_moment_stats'])
        
        return stats


class BondCalculator:
    """
    Calculate chemical bonds between atoms based on distance and chemical rules.
    """
    
    def __init__(self, bond_threshold: float = 2.0):
        self.bond_threshold = bond_threshold
        
        # Bond formation rules
        self.bond_rules = {
            'C': {'max_bonds': 4, 'can_bond_to': {'C', 'N', 'O', 'H'}},
            'N': {'max_bonds': 3, 'can_bond_to': {'C', 'H'}},
            'O': {'max_bonds': 2, 'can_bond_to': {'C', 'H'}},
            'H': {'max_bonds': 1, 'can_bond_to': {'C', 'N', 'O'}}
        }
    
    def calculate_bonds(self, atoms: List[str], coordinates: np.ndarray) -> np.ndarray:
        """
        Calculate bond adjacency matrix based on distance and chemical rules.
        
        Args:
            atoms: List of atom types
            coordinates: Array of shape (n_atoms, 3)
            
        Returns:
            Adjacency matrix of shape (n_atoms, n_atoms)
        """
        n_atoms = len(atoms)
        adjacency = np.zeros((n_atoms, n_atoms), dtype=int)
        
        # Calculate all pairwise distances
        distances = self._calculate_distances(coordinates)
        
        for i in range(n_atoms):
            atom_type = atoms[i]
            if atom_type not in self.bond_rules:
                continue
                
            # Get potential bonding partners sorted by distance
            candidates = []
            for j in range(n_atoms):
                if i != j and distances[i, j] <= self.bond_threshold:
                    if atoms[j] in self.bond_rules[atom_type]['can_bond_to']:
                        candidates.append((distances[i, j], j))
            
            # Sort by distance and take up to max_bonds
            candidates.sort()
            max_bonds = self.bond_rules[atom_type]['max_bonds']
            
            for dist, j in candidates[:max_bonds]:
                if adjacency[i, j] == 0:  # Not already bonded
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1
                    
        return adjacency
    
    def _calculate_distances(self, coordinates: np.ndarray) -> np.ndarray:
        """Calculate pairwise distances between all atoms."""
        n_atoms = coordinates.shape[0]
        distances = np.zeros((n_atoms, n_atoms))
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                distances[i, j] = dist
                distances[j, i] = dist
                
        return distances
    
    def calculate_shortest_paths(self, adjacency: np.ndarray) -> np.ndarray:
        """Calculate shortest path lengths between all pairs of atoms."""
        n_atoms = adjacency.shape[0]
        
        # Use NetworkX for shortest path calculation
        G = nx.from_numpy_array(adjacency)
        path_lengths = np.full((n_atoms, n_atoms), np.inf)
        
        for i in range(n_atoms):
            path_lengths[i, i] = 0
            if G.has_node(i):
                lengths = nx.single_source_shortest_path_length(G, i)
                for j, length in lengths.items():
                    path_lengths[i, j] = length
                    
        return path_lengths
    
    def calculate_bond_influence(self, adjacency: np.ndarray, selected_indices: List[int]) -> np.ndarray:
        """
        Calculate bond influence matrix for selected atoms.
        
        Influence = 1.0 for self-connections, 1/d^2 for connected atoms where d is shortest path length.
        """
        # Get shortest paths for the full molecule
        full_paths = self.calculate_shortest_paths(adjacency)
        
        # Extract submatrix for selected atoms
        n_selected = len(selected_indices)
        influence = np.zeros((n_selected, n_selected))
        
        for i in range(n_selected):
            for j in range(n_selected):
                idx_i, idx_j = selected_indices[i], selected_indices[j]
                
                if i == j:
                    influence[i, j] = 1.0
                else:
                    path_length = full_paths[idx_i, idx_j]
                    if path_length < np.inf and path_length > 0:
                        influence[i, j] = 1.0 / (path_length ** 2)
                    else:
                        influence[i, j] = 0.0
                        
        return influence


if __name__ == "__main__":
    # Test the data loading
    data_loader = MolecularData()
    molecules = data_loader.load_all_molecules()
    stats = data_loader.get_molecule_stats()
    
    print("\nMolecular Dataset Statistics:")
    print(f"Total molecules: {stats['total_molecules']}")
    print(f"Atom counts - mean: {stats['atom_counts'].mean():.1f}, std: {stats['atom_counts'].std():.1f}")
    print(f"V-values range: {stats['v_values'].min()} to {stats['v_values'].max()}")
    print(f"Atom type distribution: {dict(stats['atom_type_distribution'])}")
    print(f"Magnetic moment range: {stats['magnetic_moment_stats'].min():.2f} to {stats['magnetic_moment_stats'].max():.2f}")
    
    # Test bond calculation
    bond_calc = BondCalculator()
    first_mol = list(molecules.values())[0]
    adjacency = bond_calc.calculate_bonds(first_mol['atoms'], first_mol['coordinates'])
    print(f"\nFirst molecule has {adjacency.sum() // 2} bonds")
