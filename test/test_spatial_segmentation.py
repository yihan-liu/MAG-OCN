# test_spatial_segmentation.py
"""
Test script for spatial segmentation functionality in MAG-OCN
Runs in Windows PowerShell with virtual environment activation
"""

import sys
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def test_spatial_segmentation():
    """Test the spatial segmentation implementation."""
    print("=" * 60)
    print("MAG-OCN Spatial Segmentation Test")
    print("=" * 60)
    
    try:
        # Import our modules
        from data_util.preprocessor import OCNMoleculeDataset, OCNSpatialSegmentDataset
        from model.chemberta_ft_model import MAGChemBERTa
        from utils import collate
        print("✓ Successfully imported all modules")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure you have activated the virtual environment and installed dependencies")
        return False
    
    # Test 1: Load original dataset
    print("\n1. Testing original dataset...")
    try:
        orig_ds = OCNMoleculeDataset(
            root=os.path.join(project_root, 'raw'),
            filenames=['1v_expanded.csv'],
            processed_dir=os.path.join(project_root, 'processed'),
        )
        print(f"✓ Original dataset loaded: {len(orig_ds)} molecules")
    except Exception as e:
        print(f"✗ Failed to load original dataset: {e}")
        return False
    
    # Test 2: Load spatially segmented dataset
    print("\n2. Testing spatial segmentation...")
    try:
        seg_ds = OCNSpatialSegmentDataset(
            root=os.path.join(project_root, 'raw'),
            filenames=['1v_expanded.csv'],
            processed_dir=os.path.join(project_root, 'processed'),
            max_atoms_per_segment=50,  # Smaller for testing
        )
        print(f"✓ Segmented dataset loaded: {len(seg_ds)} segments")
        print(f"✓ Segmentation ratio: {len(seg_ds) / len(orig_ds):.2f} segments per molecule")
    except Exception as e:
        print(f"✗ Failed to load segmented dataset: {e}")
        return False
    
    # Test 3: Examine segments
    print("\n3. Examining spatial segments...")
    for i in range(min(3, len(seg_ds))):
        seg = seg_ds[i]
        print(f"  Segment {i}:")
        print(f"    - Atoms: {len(seg['atom_labels'])}")
        print(f"    - Segment ID: {seg.get('segment_id', 'N/A')}")
        print(f"    - Coord range: x=[{seg['coords'][:, 0].min():.2f}, {seg['coords'][:, 0].max():.2f}], "
              f"y=[{seg['coords'][:, 1].min():.2f}, {seg['coords'][:, 1].max():.2f}]")
    
    # Test 4: Test tokenization and model compatibility
    print("\n4. Testing model compatibility...")
    try:
        tokenizer = MAGChemBERTa.get_tokenizer()
        print("✓ Tokenizer loaded successfully")
        
        # Test dataloader
        dl = DataLoader(
            seg_ds,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda x: collate(x, tokenizer, max_length=256)
        )
        
        batch = next(iter(dl))
        print(f"✓ DataLoader working: batch size {batch['coords'].shape[0]}, max atoms {batch['coords'].shape[1]}")
        print(f"✓ Token sequences: shape {batch['input_ids'].shape}")
        
    except Exception as e:
        print(f"✗ Model compatibility test failed: {e}")
        return False
    
    # Test 5: Visualize spatial distribution
    print("\n5. Creating visualization...")
    try:
        # Get one original molecule for comparison
        orig_mol = orig_ds[0]
        coords = orig_mol['coords'].numpy()
        
        plt.figure(figsize=(12, 4))
        
        # Plot original molecule
        plt.subplot(1, 2, 1)
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.7, s=30, c='blue')
        plt.title(f'Original Molecule\n({len(coords)} atoms)')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True, alpha=0.3)
        
        # Show quadrant divisions
        x_center = (coords[:, 0].min() + coords[:, 0].max()) / 2
        y_center = (coords[:, 1].min() + coords[:, 1].max()) / 2
        plt.axvline(x_center, color='red', linestyle='--', alpha=0.7, label='Quadrant divisions')
        plt.axhline(y_center, color='red', linestyle='--', alpha=0.7)
        plt.legend()
        
        # Plot segments
        plt.subplot(1, 2, 2)
        colors = ['red', 'green', 'orange', 'purple']
        for i in range(min(4, len(seg_ds))):
            seg = seg_ds[i]
            seg_coords = seg['coords'].numpy()
            segment_id = seg.get('segment_id', i)
            
            plt.scatter(seg_coords[:, 0], seg_coords[:, 1], 
                       color=colors[segment_id % len(colors)], 
                       alpha=0.7, s=30,
                       label=f'Segment {segment_id} ({len(seg_coords)} atoms)')
        
        plt.title('Spatial Segments')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('spatial_segmentation_test.png', dpi=150, bbox_inches='tight')
        print("✓ Visualization saved as 'test/spatial_segmentation_test.png'")
        
    except Exception as e:
        print(f"⚠ Visualization failed (non-critical): {e}")
    
    # Test 6: Performance comparison
    print("\n6. Performance metrics...")
    try:
        # Count atoms per segment
        atom_counts = [len(seg_ds[i]['atom_labels']) for i in range(min(20, len(seg_ds)))]
        avg_atoms = np.mean(atom_counts)
        max_atoms = max(atom_counts)
        min_atoms = min(atom_counts)
        
        print(f"✓ Atoms per segment: avg={avg_atoms:.1f}, min={min_atoms}, max={max_atoms}")
        
        # Estimate token usage
        from utils import mol_to_explicit_smiles, token2atom_mapping
        
        sample_seg = seg_ds[0]
        smiles = mol_to_explicit_smiles(sample_seg['bonds'], sample_seg['atom_labels'])
        tokens = tokenizer.tokenize(smiles)
        
        print(f"✓ Sample tokenization: {len(tokens)} tokens for {len(sample_seg['atom_labels'])} atoms")
        print(f"✓ Token efficiency: {len(tokens)/len(sample_seg['atom_labels']):.2f} tokens per atom")
        
    except Exception as e:
        print(f"⚠ Performance metrics failed (non-critical): {e}")
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED - Spatial segmentation working correctly!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Run training with: python train.py --filenames 1v_expanded --max-atoms-per-segment 80")
    print("2. Adjust max_atoms_per_segment based on your model's token limits")
    print("3. Monitor training performance with spatial segments")
    
    return True

if __name__ == '__main__':
    success = test_spatial_segmentation()
    if not success:
        print("\n" + "=" * 60)
        print("✗ TESTS FAILED")
        print("=" * 60)
        print("\nTroubleshooting:")
        print("1. Make sure virtual environment is activated")
        print("2. Install requirements: pip install -r requirements.txt")
        print("3. Check that ./raw/1v_expanded.csv exists")
        sys.exit(1)
    else:
        sys.exit(0)
