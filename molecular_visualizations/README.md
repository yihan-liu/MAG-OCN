# Molecular Visualization Results

This folder contains 3D molecular visualizations generated with magnetic moment predictions using the MAG-OCN model.

## Successfully Generated Visualizations

### Basic Voltage Series (0v-7v):
- ✅ **0v_prediction.png** - 0 volt molecule (72 atoms)
- ✅ **1v_prediction.png** - 1 volt molecule (66 atoms)  
- ✅ **2v_prediction.png** - 2 volt molecule (69 atoms)
- ✅ **3v_prediction.png** - 3 volt molecule (68 atoms)
- ✅ **4v_prediction.png** - 4 volt molecule (67 atoms)
- ✅ **5v_prediction.png** - 5 volt molecule (66 atoms)
- ✅ **6v_prediction.png** - 6 volt molecule (65 atoms)
- ✅ **7v_prediction.png** - 7 volt molecule (61 atoms)

### Expanded Molecules:
- ✅ **1v_expanded_prediction.png** - 1 volt expanded molecule (264 atoms, 6 segments)

## Failed Generations

### Molecules without supported atoms (N, C, O):
- ❌ **12v.csv** - No valid atoms for model prediction
- ❌ **2v_expanded.csv** - No valid atoms for model prediction  
- ❌ **4v_expanded.csv** - No valid atoms for model prediction
- ❌ **6v_expanded.csv** - No valid atoms for model prediction
- ❌ **8v_expanded.csv** - No valid atoms for model prediction
- ❌ **10v_expanded.csv** - No valid atoms for model prediction

## Visualization Features

All successfully generated visualizations include:
- **Circular markers** for all atom types
- **Size encoding** - Marker size represents predicted magnetic moment magnitude
- **Color encoding** - Color represents signed prediction error (blue=underestimate, red=overestimate)
- **Transparent background** for clean integration
- **Reduced color range** (60% of max error) for better visibility
- **No hydrogen atoms** displayed (filtered out)
- **Bond connections** based on adjacency matrix
- **Multiple legends** - atom types, MM magnitude scale, error colorbar
- **Compact layout** (10×8 figure size)

## Model Information

- **Model**: MAG-OCN (ChemBERTa-based)
- **Checkpoint**: Epoch 85
- **Supported atoms**: N, C, O only
- **Processing**: Chunking for large molecules (50 atoms per segment)
- **Error metric**: Signed prediction error (predicted - true)

## Notes

- Expanded molecules were processed in multiple segments due to size constraints
- Some expanded molecules failed because they contain only metal atoms or other unsupported elements
- SMILES sequences were truncated when exceeding model limits (512 tokens)
- All visualizations use transparent backgrounds for better presentation integration
