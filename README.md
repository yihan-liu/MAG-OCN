# CNO magnetic moment

This repository creates a deep learning model to predict the magnetic moment of each atom in a **oxidized carbon nitride**, a family of chemicals that consist of O(xygen), C(arbon), and N(itrogen).

## Quick Start

### Windows PowerShell
```powershell
# Install dependencies
python -m pip install -r requirements.txt

# Test spatial segmentation
python test/test_spatial_segmentation.py

# Start training
python train.py --filenames 1v_expanded --epochs 10 --max-atoms-per-segment 80
```

### Windows Command Prompt
```cmd
# Install dependencies
python -m pip install -r requirements.txt

# Test spatial segmentation  
python test/test_spatial_segmentation.py

# Start training
python train.py --filenames 1v_expanded --epochs 10 --max-atoms-per-segment 80
```

### Automated Scripts
- **PowerShell**: `.\run_training.ps1`
- **Batch**: `run_training.bat`

## Spatial Segmentation

This implementation uses spatial segmentation to handle large molecules more effectively. See `SPATIAL_SEGMENTATION.md` for detailed documentation.
