# Spatial Segmentation for MAG-OCN

## Overview

This implementation introduces spatial segmentation for the MAG-OCN molecular dataset to address the token truncation issue in a more principled way. Instead of simply truncating tokens, we segment molecules spatially into coherent regions.

## Key Changes

### 1. New Dataset Class: `OCNSpatialSegmentDataset`

Located in `data_util/preprocessor.py`, this class:
- Divides each molecule into 4 spatial quadrants based on x,y coordinates
- Each quadrant becomes a separate training sample
- Preserves spatial locality of atoms within segments
- Handles variable segment sizes with configurable limits

### 2. Improved Token-to-Atom Mapping

Updated in `utils.py`:
- Better distribution of tokens across atoms in segments
- Reduced likelihood of truncation due to smaller segments
- More accurate atom-token correspondence

### 3. Enhanced Collate Function

Also in `utils.py`:
- Handles segment metadata (segment IDs, original indices)
- Optimized padding for spatial segments
- Maintains compatibility with existing model code

## Usage

### Basic Usage

```python
from data_util.preprocessor import OCNSpatialSegmentDataset

# Replace OCNMoleculeDataset with OCNSpatialSegmentDataset
ds = OCNSpatialSegmentDataset(
    root='./raw',
    filenames=['1v_expanded.csv'],
    max_atoms_per_segment=80,  # Adjust based on model capacity
    processed_dir='./processed'
)
```

### Training Script Updates

```python
# In train.py, the dataset creation becomes:
ds = OCNSpatialSegmentDataset(
    root=args.root,
    filenames=csv_filenames,
    processed_dir=args.processed_dir,
    max_atoms_per_segment=args.max_atoms_per_segment,
)
```

## Benefits

1. **Spatial Coherence**: Atoms in each training sample are spatially close, preserving local molecular structure
2. **Reduced Truncation**: Smaller segments mean fewer tokenization issues
3. **Better Data Utilization**: More training samples from the same molecular data
4. **Preserved Context**: Local chemical environments are maintained within segments
5. **Scalability**: Can handle larger molecules by segmenting appropriately

## Parameters

- `max_atoms_per_segment`: Controls the maximum number of atoms in each segment (default: 100)
- `threshold`: Distance threshold for bond detection (default: 2.0)
- Standard dataset parameters remain the same

## Testing

Run the test script to verify spatial segmentation:

```powershell
python test/test_spatial_segmentation.py
```

This will:
- Compare original vs segmented datasets
- Visualize spatial distribution
- Test tokenization and data loading
- Generate diagnostic plots in `test/spatial_segmentation_test.png`

## Example

```powershell
python train.py --filenames 1v_expanded --max-atoms-per-segment 80
```

This demonstrates training with spatial segmentation.

## Notes

- Segments are cached for efficiency (separate cache from original dataset)
- Each molecule typically produces 2-4 segments depending on atom distribution
- Empty quadrants are automatically skipped
- Original atom indices are preserved for reconstruction if needed

## Migration from Original Dataset

Simply replace:
```python
from data_util.preprocessor import OCNMoleculeDataset
ds = OCNMoleculeDataset(...)
```

With:
```python
from data_util.preprocessor import OCNSpatialSegmentDataset
ds = OCNSpatialSegmentDataset(...)
```

All other code remains compatible.
