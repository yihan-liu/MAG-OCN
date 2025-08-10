#!/usr/bin/env python3
"""
Simple 3D Molecular Visualization (Structure Only)

This is a simplified version that focuses on visualization without
model predictions, to demonstrate the core visualization capabilities.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')

class MolecularVisualizationError(Exception):
    """Custom exception for molecular visualization errors."""
    pass

class SimpleMolecularVisualizer:
    """Handles 3D visualization of molecules."""
    
    # Atom type to shape mapping (all circles now)
    ATOM_SHAPES = {
        'H': 'o',   # circle
        'C': 'o',   # circle
        'N': 'o',   # circle
        'O': 'o',   # circle
        'S': 'o',   # circle
        'P': 'o',   # circle
        'F': 'o',   # circle
        'Cl': 'o',  # circle
        'Br': 'o',  # circle
        'I': 'o',   # circle
    }
    
    # Atom type to color mapping
    ATOM_COLORS = {
        'H': '#FFFFFF',   # white
        'C': '#000000',   # black
        'N': '#0000FF',   # blue
        'O': '#FF0000',   # red
        'S': '#FFFF00',   # yellow
        'P': '#FF8000',   # orange
        'F': '#00FF00',   # green
        'Cl': '#00FFFF',  # cyan
        'Br': '#800080',  # purple
        'I': '#800000',   # maroon
    }
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """Initialize the visualizer with plot settings."""
        self.figsize = figsize
        
    def plot_molecule_3d(self, mol_data: pd.DataFrame, save_path: str = None, 
                        title: str = None) -> plt.Figure:
        """
        Create 3D visualization of molecular structure.
        
        Args:
            mol_data: DataFrame with molecular data
            save_path: Path to save the plot
            title: Custom title for the plot
            
        Returns:
            matplotlib Figure object
        """
        # Extract data and filter out hydrogen atoms
        all_coordinates = mol_data[['X', 'Y', 'Z']].values
        all_atom_symbols = [atom.split()[0][0] for atom in mol_data['ATOM']]
        all_true_moments = mol_data['MAGNETIC_MOMENT'].values
        
        # Filter out hydrogen atoms
        coordinates_list = []
        atom_symbols = []
        true_moments_list = []
        
        for i, symbol in enumerate(all_atom_symbols):
            if symbol != 'H':  # Skip hydrogen atoms
                coordinates_list.append(all_coordinates[i])
                atom_symbols.append(symbol)
                true_moments_list.append(all_true_moments[i])
        
        coordinates = np.array(coordinates_list)
        true_moments = np.array(true_moments_list)
        
        # Center coordinates
        coordinates = coordinates - coordinates.mean(axis=0, keepdims=True)
        
        # Build bonds using the same rules as the dataset
        from data_util.graph_utils import _build_adjacency
        bonds = _build_adjacency(coordinates, atom_symbols, threshold=2.0)
        
        # Create figure
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Make background transparent
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        # Plot 3D molecular structure
        self._plot_3d_structure(ax, coordinates, atom_symbols, true_moments, bonds)
        
        # Set title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title('3D Molecular Structure', fontsize=14, fontweight='bold')
        
        # Add legend for atom types
        self._add_atom_legend(ax, atom_symbols)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
            print(f"Plot saved to: {save_path}")
        
        return fig
    
    def _plot_3d_structure(self, ax, coordinates, atom_symbols, true_moments, 
                          bonds=None):
        """Plot the 3D molecular structure with bonds and clean styling."""
        
        # Plot bonds first (so they appear behind atoms)
        if bonds is not None:
            self._plot_bonds(ax, coordinates, bonds)
        
        # Calculate size scaling parameters for true magnetic moments
        max_true_moment = np.max(np.abs(true_moments)) if len(true_moments) > 0 else 1.0
        min_size, max_size = 50, 200  # Size range for markers
        
        # Plot atoms with shapes by type, colors by type, size by true MM magnitude
        for i, (coord, symbol) in enumerate(zip(coordinates, atom_symbols)):
            x, y, z = coord
            
            # Get atom properties
            shape = self.ATOM_SHAPES.get(symbol, 'o')
            color = self.ATOM_COLORS.get(symbol, '#808080')
            
            # Size based on true magnetic moment magnitude
            true_moment_mag = abs(true_moments[i])
            if max_true_moment > 0:
                size_ratio = true_moment_mag / max_true_moment
                marker_size = min_size + (max_size - min_size) * size_ratio
            else:
                marker_size = min_size
            
            # Plot atom
            ax.scatter(x, y, z, c=color, marker=shape, s=marker_size, alpha=0.8, 
                      edgecolors='none', linewidth=0)
            
            # Add atom label
            ax.text(x, y, z + 0.3, f'{symbol}{i+1}', fontsize=8, ha='center')
        
        # Remove all axis styling and make plot clean
        self._clean_3d_plot(ax, coordinates)
    
    def _plot_bonds(self, ax, coordinates, bonds):
        """Plot bonds between atoms as lines."""
        bonds = np.array(bonds)
        n_atoms = len(coordinates)
        
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):  # Only plot each bond once
                if bonds[i, j] > 0:  # Bond exists
                    x_coords = [coordinates[i][0], coordinates[j][0]]
                    y_coords = [coordinates[i][1], coordinates[j][1]]
                    z_coords = [coordinates[i][2], coordinates[j][2]]
                    
                    ax.plot(x_coords, y_coords, z_coords, 
                           color='gray', linewidth=1.5, alpha=0.6)
    
    def _clean_3d_plot(self, ax, coordinates):
        """Remove all axis ticks, labels, and backgrounds to create a clean 3D plot."""
        # Remove axis labels and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Remove axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        
        # Remove grid
        ax.grid(False)
        
        # Remove panes (background walls)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Make pane edges transparent
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        
        # Remove axis lines
        ax.xaxis.line.set_color('none')
        ax.yaxis.line.set_color('none')
        ax.zaxis.line.set_color('none')
        
        # Make the background transparent
        ax.xaxis.pane.set_alpha(0)
        ax.yaxis.pane.set_alpha(0)
        ax.zaxis.pane.set_alpha(0)
        
        # Set equal aspect ratio for proper molecule representation
        max_range = np.array([coordinates[:,0].max()-coordinates[:,0].min(),
                             coordinates[:,1].max()-coordinates[:,1].min(),
                             coordinates[:,2].max()-coordinates[:,2].min()]).max() / 2.0
        
        mid_x = (coordinates[:,0].max()+coordinates[:,0].min()) * 0.5
        mid_y = (coordinates[:,1].max()+coordinates[:,1].min()) * 0.5
        mid_z = (coordinates[:,2].max()+coordinates[:,2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    def _add_atom_legend(self, ax, atom_symbols):
        """Add legend for atom types (showing shape and color for structure-only mode)."""
        unique_symbols = list(set(atom_symbols))
        legend_elements = []
        
        for symbol in sorted(unique_symbols):
            shape = self.ATOM_SHAPES.get(symbol, 'o')
            color = self.ATOM_COLORS.get(symbol, '#808080')
            
            # Create legend entry
            legend_elements.append(plt.Line2D([0], [0], marker=shape, color='w', 
                                            markerfacecolor=color, markersize=10,
                                            markeredgecolor='none', label=symbol))  # Remove borders
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

def load_molecule_data(csv_path: str) -> pd.DataFrame:
    """Load molecule data from CSV file."""
    if not os.path.exists(csv_path):
        raise MolecularVisualizationError(f"CSV file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['ATOM', 'X', 'Y', 'Z', 'MAGNETIC_MOMENT']
        
        if not all(col in df.columns for col in required_columns):
            raise MolecularVisualizationError(f"CSV must contain columns: {required_columns}")
        
        return df
    except Exception as e:
        raise MolecularVisualizationError(f"Error loading CSV: {str(e)}")

def main():
    """Main function to run molecular visualization."""
    parser = argparse.ArgumentParser(description='Simple 3D Molecular Structure Visualization')
    parser.add_argument('--molecule', type=str, required=True,
                       help='Path to molecule CSV file or name of file in raw/ directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for visualization (default: auto-generated)')
    parser.add_argument('--title', type=str, default=None,
                       help='Custom title for the plot')
    
    args = parser.parse_args()
    
    try:
        # Resolve molecule path
        if os.path.exists(args.molecule):
            mol_path = args.molecule
        else:
            # Try in raw/ directory
            mol_path = os.path.join('raw', args.molecule)
            if not mol_path.endswith('.csv'):
                mol_path += '.csv'
        
        # Load molecule data
        print(f"🧬 Loading molecule data from: {mol_path}")
        mol_data = load_molecule_data(mol_path)
        print(f"   Loaded molecule with {len(mol_data)} atoms")
        
        # Show atom type distribution
        atom_symbols = [atom.split()[0][0] for atom in mol_data['ATOM']]
        atom_counts = {}
        for symbol in atom_symbols:
            atom_counts[symbol] = atom_counts.get(symbol, 0) + 1
        print(f"   Atom types: {dict(sorted(atom_counts.items()))}")
        
        # Initialize visualizer
        visualizer = SimpleMolecularVisualizer()
        
        # Visualize structure
        print("🎨 Creating 3D visualization...")
        fig = visualizer.plot_molecule_3d(
            mol_data,
            title=args.title or f"Molecular Structure - {os.path.basename(mol_path)}"
        )
        
        # Save plot
        if args.output:
            output_path = args.output
        else:
            mol_name = os.path.splitext(os.path.basename(mol_path))[0]
            output_path = f"simple_molecule_viz_{mol_name}.png"
        
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to: {output_path}")
        
        # Show plot
        plt.show()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
