#!/usr/bin/env python3
"""
3D Molecular Visualization with Magnetic Moment Predictions

This script provides comprehensive 3D visualization capabilities for molecules
with magnetic moment predictions from the trained MAG-OCN model.

Features:
- Interactive 3D plotting of molecular structures
- Shape-coded atom types with legend
- Magnetic moment prediction visualization with arrows
- Color-coded prediction errors
- Sup        # Handle predictions - expand to display atom list if available
        if predictions is not None:
            if len(predictions) != len(model_atom_symbols):
                raise ValueError(f"Predictions length ({len(predictions)}) doesn't match model atoms ({len(model_atom_symbols)})")
            
            # Create display predictions array (predictions only for model atoms within display atoms)
            display_predictions = np.full(len(display_atom_symbols), np.nan)
            for i, model_idx in enumerate(model_atom_indices):
                display_predictions[model_idx] = predictions[i]
            
            # Calculate signed errors only for model atoms
            model_errors = predictions - model_true_moments
            max_abs_error = np.max(np.abs(model_errors)) if len(model_errors) > 0 else 1.0
        else:
            display_predictions = None
            max_abs_error = Noneolecules through spatial segmentation
- Batch processing for molecules exceeding model limits
- Model checkpoint loading and evaluation

Author: MAG-OCN Project
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from data_util.preprocessor import OCNSpatialSegmentDataset
from model.chemberta_ft_model import MAGChemBERTa
from utils import collate, token2atom_mapping
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better plots - use fallback if seaborn not available
try:
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
except ImportError:
    plt.style.use('default')
    print("Note: seaborn not available, using default matplotlib style")

class MolecularVisualizationError(Exception):
    """Custom exception for molecular visualization errors."""
    pass

class MoleculePredictor:
    """Handles model loading and prediction for molecular magnetic moments."""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """
        Initialize the predictor with a trained model checkpoint.
        
        Args:
            checkpoint_path: Path to the model checkpoint
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.device = self._setup_device(device)
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        # Initialize tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup the appropriate device for inference."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_model(self, checkpoint_path: str) -> MAGChemBERTa:
        """Load the trained model from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise MolecularVisualizationError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize model with default parameters
        model = MAGChemBERTa()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"Model loaded successfully. Epoch: {checkpoint.get('epoch', 'unknown')}")
        return model
    
    def predict_molecule(self, mol_data: pd.DataFrame, max_atoms_per_segment: int = 50) -> Tuple[np.ndarray, Dict]:
        """
        Predict magnetic moments for a molecule, handling large molecules through segmentation.
        
        Args:
            mol_data: DataFrame with columns [ATOM, X, Y, Z, MAGNETIC_MOMENT]
            max_atoms_per_segment: Maximum atoms per spatial segment
            
        Returns:
            predictions: Array of predicted magnetic moments
            metadata: Dictionary with prediction metadata
        """
        n_atoms = len(mol_data)
        
        if n_atoms <= max_atoms_per_segment:
            # Process as single segment
            return self._predict_single_segment(mol_data)
        else:
            # Process with simple chunking for now
            return self._predict_with_chunking(mol_data, max_atoms_per_segment)
    
    def _predict_single_segment(self, mol_data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Predict magnetic moments for a single molecular segment."""
        try:
            # Create temporary dataset for this molecule
            temp_dataset = SingleMoleculeDataset(mol_data)
            
            # Get data sample
            sample = temp_dataset[0]
            
            # Create batch
            batch = collate([sample], self.tokenizer)
            
            # Move to device (only move tensors, leave lists as-is)
            device_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    device_batch[k] = v.to(self.device)
                else:
                    device_batch[k] = v
            
            # Only pass arguments the model expects
            model_args = {
                'input_ids': device_batch['input_ids'],
                'attention_mask': device_batch['attention_mask'],
                'coords': device_batch['coords'],
                'token2atom': device_batch['token2atom'],
                'mask': device_batch['mask']
            }
            
            # Predict
            with torch.no_grad():
                predictions = self.model(**model_args)
                
            # Extract predictions for actual atoms (not padding)
            n_atoms = len(sample['atom_labels'])
            pred_reduced = predictions[0, :n_atoms].cpu().numpy()
            
            # Recover original magnetic moments from reduced scale
            from data_util.preprocessor import OCNMoleculeDataset
            pred_original = OCNMoleculeDataset._recover_mm(pred_reduced)
            
            metadata = {
                'method': 'single_segment',
                'n_segments': 1,
                'total_atoms': n_atoms,
                'device': str(self.device)
            }
            
            return pred_original, metadata
            
        except Exception as e:
            raise MolecularVisualizationError(f"Prediction failed: {str(e)}")
    
    def _predict_with_segmentation(self, mol_data: pd.DataFrame, max_atoms: int) -> Tuple[np.ndarray, Dict]:
        """Predict magnetic moments using spatial segmentation for large molecules."""
        try:
            # Create spatial segmentation dataset
            temp_csv = 'temp_molecule.csv'
            mol_data.to_csv(temp_csv, index=False)
            
            dataset = OCNSpatialSegmentDataset(
                root='.',
                filenames=[temp_csv],
                max_atoms_per_segment=max_atoms
            )
            
            # Collect all predictions
            all_predictions = []
            atom_indices = []
            
            print(f"Processing {len(dataset)} spatial segments...")
            for i in tqdm(range(len(dataset)), desc="Predicting segments"):
                sample = dataset[i]
                
                # Create batch
                batch = collate([sample], self.tokenizer)
                
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Predict
                with torch.no_grad():
                    predictions = self.model(batch)
                
                # Extract segment data - number of atoms in this segment
                n_atoms_in_segment = len(sample['atom_labels'])
                
                # Store predictions and indices
                pred_segment = predictions[0, :n_atoms_in_segment].cpu().numpy()
                all_predictions.extend(pred_segment)
                
                # Get original indices if available, otherwise use sequential
                if 'original_indices' in sample:
                    atom_indices.extend(sample['original_indices'])
                else:
                    # Fallback: use sequential indices
                    start_idx = i * max_atoms
                    atom_indices.extend(range(start_idx, start_idx + n_atoms_in_segment))
            
            # Reorder predictions to match original atom order
            full_predictions = np.zeros(len(mol_data))
            for idx, pred in zip(atom_indices, all_predictions):
                if idx < len(mol_data):  # Safety check
                    full_predictions[idx] = pred
            
            # Clean up temp file
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
            
            metadata = {
                'method': 'spatial_segmentation',
                'n_segments': len(dataset),
                'total_atoms': len(mol_data),
                'max_atoms_per_segment': max_atoms,
                'device': str(self.device)
            }
            
            return full_predictions, metadata
            
        except Exception as e:
            if os.path.exists('temp_molecule.csv'):
                os.remove('temp_molecule.csv')
            raise MolecularVisualizationError(f"Segmented prediction failed: {str(e)}")
    
    def _predict_with_chunking(self, mol_data: pd.DataFrame, max_atoms: int) -> Tuple[np.ndarray, Dict]:
        """Predict magnetic moments using simple chunking for large molecules."""
        try:
            n_atoms = len(mol_data)
            all_predictions = []
            
            print(f"Processing molecule in chunks of {max_atoms} atoms...")
            
            # Process in chunks
            for start_idx in tqdm(range(0, n_atoms, max_atoms), desc="Processing chunks"):
                end_idx = min(start_idx + max_atoms, n_atoms)
                chunk_data = mol_data.iloc[start_idx:end_idx].copy()
                
                # Predict for this chunk
                chunk_pred, _ = self._predict_single_segment(chunk_data)
                all_predictions.extend(chunk_pred)
            
            metadata = {
                'method': 'simple_chunking',
                'n_segments': (n_atoms + max_atoms - 1) // max_atoms,
                'total_atoms': n_atoms,
                'max_atoms_per_segment': max_atoms,
                'device': str(self.device)
            }
            
            return np.array(all_predictions), metadata
            
        except Exception as e:
            raise MolecularVisualizationError(f"Chunked prediction failed: {str(e)}")


class SingleMoleculeDataset:
    """Simple dataset wrapper for single molecule prediction."""
    
    def __init__(self, mol_data: pd.DataFrame):
        self.mol_data = mol_data
        
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        from data_util.graph_utils import _build_adjacency
        from data_util.atoms_encoding import ATOM_DICT
        
        # Extract atom information and filter to only include valid atoms
        all_atom_symbols = [atom.split()[0][0] for atom in self.mol_data['ATOM']]  # Extract element symbol
        all_coordinates = self.mol_data[['X', 'Y', 'Z']].values
        all_magnetic_moments = self.mol_data['MAGNETIC_MOMENT'].values
        
        # Filter to only include atoms in the model's vocabulary (exclude H)
        atom_symbols = []
        coordinates_list = []
        magnetic_moments_list = []
        
        for i, symbol in enumerate(all_atom_symbols):
            if symbol in ATOM_DICT:  # Only include N, C, O
                atom_symbols.append(symbol)
                coordinates_list.append(all_coordinates[i])
                magnetic_moments_list.append(all_magnetic_moments[i])
        
        if len(atom_symbols) == 0:
            raise ValueError("No valid atoms found in molecule (only N, C, O are supported)")
        
        coordinates = np.array(coordinates_list)
        magnetic_moments = np.array(magnetic_moments_list)
        
        # Center coordinates
        coordinates = coordinates - coordinates.mean(axis=0, keepdims=True)
        
        # Build adjacency matrix using the same rules as the dataset
        adj_matrix = _build_adjacency(coordinates, atom_symbols, threshold=2.0)
        
        # Apply the same magnetic moment reduction as the original dataset
        from data_util.preprocessor import OCNMoleculeDataset
        magnetic_moments_reduced = np.array([OCNMoleculeDataset._reduce_mm(mm) for mm in magnetic_moments])
        
        # Convert to the format expected by the model
        return {
            'atom_labels': atom_symbols,
            'coords': torch.FloatTensor(coordinates),
            'bonds': torch.FloatTensor(adj_matrix),
            'mm_reduced': torch.FloatTensor(magnetic_moments_reduced),
            'mm_original': torch.FloatTensor(magnetic_moments),
            'v_value': 0  # Placeholder
        }


class MolecularVisualizer:
    """Handles 3D visualization of molecules with magnetic moment predictions."""
    
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
        
    def plot_molecule_3d(self, mol_data: pd.DataFrame, predictions: np.ndarray = None, 
                        metadata: Dict = None, save_path: str = None, 
                        title: str = None) -> plt.Figure:
        """
        Create comprehensive 3D visualization of molecular structure with predictions.
        
        Args:
            mol_data: DataFrame with molecular data
            predictions: Array of predicted magnetic moments
            metadata: Prediction metadata dictionary
            save_path: Path to save the plot
            title: Custom title for the plot
            
        Returns:
            matplotlib Figure object
        """
        from data_util.graph_utils import _build_adjacency
        from data_util.atoms_encoding import ATOM_DICT
        
        # Extract data
        all_coordinates = mol_data[['X', 'Y', 'Z']].values
        all_atom_symbols = [atom.split()[0][0] for atom in mol_data['ATOM']]
        all_true_moments = mol_data['MAGNETIC_MOMENT'].values
        
        # Filter out hydrogen atoms for both visualization and predictions
        display_atom_indices = []
        display_atom_symbols = []
        display_coordinates_list = []
        display_true_moments_list = []
        
        model_atom_indices = []  # Indices within the display arrays
        model_atom_symbols = []
        model_coordinates_list = []
        model_true_moments_list = []
        
        display_idx = 0
        for i, symbol in enumerate(all_atom_symbols):
            if symbol != 'H':  # Skip hydrogen atoms for display
                display_atom_indices.append(i)
                display_atom_symbols.append(symbol)
                display_coordinates_list.append(all_coordinates[i])
                display_true_moments_list.append(all_true_moments[i])
                
                if symbol in ATOM_DICT:  # Also include in model if N, C, O
                    model_atom_indices.append(display_idx)
                    model_atom_symbols.append(symbol)
                    model_coordinates_list.append(all_coordinates[i])
                    model_true_moments_list.append(all_true_moments[i])
                
                display_idx += 1
        
        if len(model_atom_symbols) == 0:
            raise ValueError("No valid atoms found for model (only N, C, O are supported)")
        
        display_coordinates = np.array(display_coordinates_list)
        display_true_moments = np.array(display_true_moments_list)
        model_coordinates = np.array(model_coordinates_list)
        model_true_moments = np.array(model_true_moments_list)
        
        # Center coordinates
        display_coordinates = display_coordinates - display_coordinates.mean(axis=0, keepdims=True)
        
        # Build bonds using display atoms (no hydrogen)
        bonds = _build_adjacency(display_coordinates, display_atom_symbols, threshold=2.0)
        
        # Handle predictions - expand to display atom list if available
        if predictions is not None:
            if len(predictions) != len(model_atom_symbols):
                raise ValueError(f"Predictions length ({len(predictions)}) doesn't match model atoms ({len(model_atom_symbols)})")
            
            # Create display predictions array (predictions only for model atoms within display atoms)
            display_predictions = np.full(len(display_atom_symbols), np.nan)
            for i, model_idx in enumerate(model_atom_indices):
                display_predictions[model_idx] = predictions[i]
            
            # Calculate signed errors only for model atoms
            model_errors = predictions - model_true_moments
            max_abs_error = np.max(np.abs(model_errors)) if len(model_errors) > 0 else 1.0
        else:
            display_predictions = None
            max_abs_error = None
        
        # Create figure with single subplot (always just the 3D structure)
        fig = plt.figure(figsize=self.figsize)
        ax1 = fig.add_subplot(111, projection='3d')
        
        # Make background transparent
        fig.patch.set_alpha(0.0)
        ax1.patch.set_alpha(0.0)
        
        # Plot 3D molecular structure (using display atoms, no hydrogen)
        self._plot_3d_structure(ax1, display_coordinates, display_atom_symbols, display_true_moments, 
                               bonds, display_predictions, model_atom_indices, max_abs_error)
        
        # Set title
        if title:
            ax1.set_title(title, fontsize=14, fontweight='bold')
        elif predictions is not None:
            method = metadata.get('method', 'unknown') if metadata else 'unknown'
            ax1.set_title(f'3D Molecular Structure with Predictions\nMethod: {method}', 
                         fontsize=14, fontweight='bold')
        else:
            ax1.set_title('3D Molecular Structure', fontsize=14, fontweight='bold')
        
        # Add legend for atom types
        self._add_atom_legend(ax1, display_atom_symbols)
        
        # Add size legend for predicted magnetic moments if predictions available
        if predictions is not None:
            self._add_size_legend(ax1, predictions)
        
        # Add color bar for prediction errors if predictions available
        if predictions is not None:
            # Use reduced range for colorbar to match visualization
            reduced_max_error = max_abs_error * 0.6
            self._add_error_colorbar(fig, ax1, model_errors, reduced_max_error)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
            print(f"Plot saved to: {save_path}")
        
        return fig
    
    def _plot_3d_structure(self, ax, coordinates, atom_symbols, true_moments, 
                          bonds=None, display_predictions=None, model_atom_indices=None, 
                          max_abs_error=None):
        """Plot the 3D molecular structure with bonds and error-based coloring."""
        
        # Plot bonds first (so they appear behind atoms)
        if bonds is not None:
            self._plot_bonds(ax, coordinates, bonds)
        
        # Calculate size scaling parameters for predicted magnetic moments
        if display_predictions is not None:
            # Get predicted moments for atoms that have predictions
            pred_moments = []
            for i in range(len(coordinates)):
                if (model_atom_indices is not None and i in model_atom_indices 
                    and not np.isnan(display_predictions[i])):
                    pred_moments.append(abs(display_predictions[i]))
            
            if pred_moments:
                max_pred_moment = max(pred_moments)
                min_size, max_size = 50, 200  # Size range for markers
            else:
                max_pred_moment = 1.0
                min_size, max_size = 100, 100  # Default size if no predictions
        else:
            # For structure-only mode, use true magnetic moments for sizing
            true_moments_abs = [abs(mm) for mm in true_moments]
            max_pred_moment = max(true_moments_abs) if true_moments_abs else 1.0
            min_size, max_size = 50, 200
        
        # Plot atoms with shapes by type, colors by error, size by predicted MM
        for i, (coord, symbol) in enumerate(zip(coordinates, atom_symbols)):
            x, y, z = coord
            
            # Get atom shape (determined by type)
            shape = self.ATOM_SHAPES.get(symbol, 'o')
            
            # Determine marker size based on predicted magnetic moment magnitude
            if (display_predictions is not None and model_atom_indices is not None 
                and i in model_atom_indices and not np.isnan(display_predictions[i])):
                # This atom has a prediction - size based on predicted moment magnitude
                pred_moment_mag = abs(display_predictions[i])
                if max_pred_moment > 0:
                    size_ratio = pred_moment_mag / max_pred_moment
                    marker_size = min_size + (max_size - min_size) * size_ratio
                else:
                    marker_size = min_size
                
                # Color by signed prediction error (predicted - true)
                signed_error = display_predictions[i] - true_moments[i]
                # Reduce color range for better visibility (use 60% of max error range)
                reduced_max_error = max_abs_error * 0.6 if max_abs_error > 0 else 1.0
                # Normalize to [-1, 1] range for color mapping
                normalized_error = signed_error / reduced_max_error if reduced_max_error > 0 else 0
                # Clamp to [-1, 1] range
                normalized_error = np.clip(normalized_error, -1, 1)
                # Map to [0, 1] for colormap (0.5 = no error, 0 = negative error, 1 = positive error)
                color_value = (normalized_error + 1) / 2
                # Use coolwarm colormap: blue (underestimation) to red (overestimation)
                color = plt.cm.coolwarm(color_value)
            else:
                # No prediction for this atom - use true MM for size and atom type color
                true_moment_mag = abs(true_moments[i])
                if max_pred_moment > 0:
                    size_ratio = true_moment_mag / max_pred_moment
                    marker_size = min_size + (max_size - min_size) * size_ratio
                else:
                    marker_size = min_size
                color = self.ATOM_COLORS.get(symbol, '#808080')
            
            # Plot atom
            ax.scatter(x, y, z, c=[color], marker=shape, s=marker_size, 
                      edgecolors='none', linewidth=0, alpha=0.8)
            
            # Add atom label
            ax.text(x, y, z + 0.2, f'{symbol}{i+1}', fontsize=8, ha='center')
        
        # Remove all axis styling and make plot clean
        self._clean_3d_plot(ax, coordinates)
    
    def _add_atom_legend(self, ax, atom_symbols):
        """Add legend for atom types (colors only since all shapes are circles)."""
        unique_symbols = list(set(atom_symbols))
        legend_elements = []
        
        for symbol in sorted(unique_symbols):
            color = self.ATOM_COLORS.get(symbol, '#808080')
            # Use atom type colors for legend since all shapes are circles
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=10,
                                            markeredgecolor='none', label=f'{symbol}'))
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    def _add_size_legend(self, ax, predictions):
        """Add legend showing marker size scale for predicted magnetic moments."""
        # Calculate size ranges
        pred_moments = predictions[~np.isnan(predictions)]
        if len(pred_moments) == 0:
            return
        
        max_pred = np.max(np.abs(pred_moments))
        min_pred = np.min(np.abs(pred_moments))
        
        # Create size legend with 3 representative sizes
        if max_pred > min_pred:
            legend_values = [min_pred, (min_pred + max_pred) / 2, max_pred]
        else:
            legend_values = [max_pred]
        
        min_size, max_size = 50, 200
        legend_elements = []
        
        for i, val in enumerate(legend_values):
            if max_pred > 0:
                size_ratio = abs(val) / max_pred
                marker_size = min_size + (max_size - min_size) * size_ratio
            else:
                marker_size = min_size
            
            # Scale marker size for legend display
            display_size = max(marker_size / 15, 4)  # Scale down for legend
            
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor='gray', markersize=display_size,
                                            markeredgecolor='none', 
                                            label=f'|MM|={abs(val):.2f}'))
        
        # Add the size legend in the upper right
        size_legend = ax.legend(handles=legend_elements, loc='upper right', 
                              bbox_to_anchor=(1, 1), title='Predicted |MM| (size)')
        size_legend.get_title().set_fontsize(10)
        
        # Make sure the atom legend is still visible by adjusting the size legend
        ax.add_artist(size_legend)
    
    def _plot_error_analysis(self, ax, true_moments, predictions, errors, metadata):
        """Plot error analysis and statistics."""
        # Scatter plot: true vs predicted
        ax.scatter(true_moments, predictions, alpha=0.7, s=50)
        
        # Perfect prediction line
        min_val = min(np.min(true_moments), np.min(predictions))
        max_val = max(np.max(true_moments), np.max(predictions))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, 
                label='Perfect Prediction')
        
        # Labels and formatting
        ax.set_xlabel('True Magnetic Moment', fontsize=12)
        ax.set_ylabel('Predicted Magnetic Moment', fontsize=12)
        ax.set_title('Prediction Accuracy', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics text (use absolute errors for MAE, signed errors for RMSE)
        mae = np.mean(np.abs(errors))  # Mean Absolute Error
        rmse = np.sqrt(np.mean(errors**2))  # Root Mean Square Error
        r2 = np.corrcoef(true_moments, predictions)[0, 1]**2
        
        stats_text = f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}'
        if metadata:
            if 'n_segments' in metadata:
                stats_text += f'\nSegments: {metadata["n_segments"]}'
            if 'total_atoms' in metadata:
                stats_text += f'\nAtoms: {metadata["total_atoms"]}'
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
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
    
    def _add_error_colorbar(self, fig, ax, errors, max_abs_error):
        """Add a color bar showing the signed prediction error scale."""
        import matplotlib.cm as cm
        import matplotlib.colors as colors
        
        # Create a mappable object for the color bar with symmetric range
        norm = colors.Normalize(vmin=-max_abs_error, vmax=max_abs_error)
        mappable = cm.ScalarMappable(norm=norm, cmap='coolwarm')
        mappable.set_array([])
        
        # Add color bar
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.8, pad=0.1)
        cbar.set_label('Prediction Error (Pred - True)\nBlue: Underestimation, Red: Overestimation', 
                      rotation=270, labelpad=20)
        cbar.ax.tick_params(labelsize=10)


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
    parser = argparse.ArgumentParser(description='3D Molecular Visualization with Magnetic Moment Predictions')
    parser.add_argument('--molecule', type=str, required=True,
                       help='Path to molecule CSV file or name of file in raw/ directory')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_checkpoint.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for visualization (default: auto-generated)')
    parser.add_argument('--max-atoms-per-segment', type=int, default=50,
                       help='Maximum atoms per spatial segment for large molecules')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device for inference')
    parser.add_argument('--no-prediction', action='store_true',
                       help='Only visualize structure without predictions')
    
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
        print(f"Loading molecule data from: {mol_path}")
        mol_data = load_molecule_data(mol_path)
        print(f"Loaded molecule with {len(mol_data)} atoms")
        
        # Initialize visualizer
        visualizer = MolecularVisualizer()
        
        if args.no_prediction:
            # Just visualize structure
            print("Visualizing molecular structure only...")
            fig = visualizer.plot_molecule_3d(
                mol_data,
                title="Molecular Structure"
            )
            
        else:
            # Load model and predict
            predictor = MoleculePredictor(args.checkpoint, args.device)
            
            print("Predicting magnetic moments...")
            predictions, metadata = predictor.predict_molecule(
                mol_data, args.max_atoms_per_segment
            )
            
            print(f"Prediction completed using {metadata['method']}")
            if 'n_segments' in metadata:
                print(f"Used {metadata['n_segments']} spatial segments")
            
            # Visualize with predictions
            fig = visualizer.plot_molecule_3d(
                mol_data,
                predictions=predictions,
                metadata=metadata
            )
        
        # Save plot
        if args.output:
            output_path = args.output
        else:
            mol_name = os.path.splitext(os.path.basename(mol_path))[0]
            output_path = f"molecule_viz_{mol_name}.png"
        
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        
        # Show plot
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
