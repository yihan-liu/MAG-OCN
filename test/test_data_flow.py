# test_data_flow.py

import sys
import os
# Add parent directory to path so we can import from the main package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from data_util.preprocessor import OCNMoleculeDataset
from model.chemberta_ft_model import ChemBERTaWithCoords
from utils import mol_to_explicit_smiles, token2atom_mapping, collate

def test_single_molecule():
    """Test data flow for a single molecule from the dataset."""
    print("=== Testing Single Molecule Data Flow ===\n")
    
    # Load a small dataset (just one file)
    ds = OCNMoleculeDataset(
        root='./raw',
        filenames=['5v.csv'],  # Start with one file
        processed_dir='./processed',
        augmentations=None,
    )
    
    if len(ds) == 0:
        print("ERROR: No molecules loaded from dataset!")
        return
    
    # Get first molecule
    mol = ds[0]
    print(f"Dataset loaded {len(ds)} molecules")
    print(f"First molecule data keys: {list(mol.keys())}")
    print()
    
    # Inspect molecule structure
    print("=== Molecule Structure ===")
    print(f"Atom labels: {mol['atom_labels']}")
    print(f"Number of atoms: {len(mol['atom_labels'])}")
    print(f"Coordinates shape: {mol['coords'].shape}")
    print(f"Bonds shape: {mol['bonds'].shape}")
    print(f"MM reduced shape: {mol['mm_reduced'].shape}")
    print(f"MM original shape: {mol['mm_original'].shape}")
    print(f"V-value: {mol['v_value']}")
    print()
    
    # Test SMILES conversion
    print("=== SMILES Conversion ===")
    smiles = mol_to_explicit_smiles(mol['bonds'], mol['atom_labels'])
    print(f"Generated SMILES: '{smiles}'")
    print()
    
    # Test tokenization
    print("=== Tokenization ===")
    tokenizer = ChemBERTaWithCoords.get_tokenizer()
    tokens = tokenizer.tokenize(smiles)
    print(f"Tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
    
    # Full tokenization (with special tokens)
    encoded = tokenizer(smiles, return_tensors='pt')
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids}")
    print()
    
    # Test token2atom mapping
    print("=== Token-to-Atom Mapping ===")
    n_atoms = len(mol['atom_labels'])
    token2atom = token2atom_mapping(smiles, tokenizer, n_atoms)
    print(f"Token2atom mapping: {token2atom}")
    print(f"Mapping length: {len(token2atom)}")
    print(f"Expected length (tokens + 2): {len(tokens) + 2}")
    
    # Detailed mapping analysis
    print("\n=== Detailed Token Analysis ===")
    full_tokens = ['[CLS]'] + tokens + ['[SEP]']
    for i, (token, atom_idx) in enumerate(zip(full_tokens[:len(token2atom)], token2atom)):
        print(f"  Token {i:2d}: '{token:8s}' → Atom {atom_idx}")
    print()
    
    return mol, smiles, token2atom

def test_batch_collate():
    """Test the collate function with a small batch."""
    print("=== Testing Batch Collate Function ===\n")
    
    # Load dataset
    ds = OCNMoleculeDataset(
        root='./raw',
        filenames=['5v.csv'],
        processed_dir='./processed',
        augmentations=None,
    )
    
    # Get a small batch (first 2 molecules)
    batch_size = min(2, len(ds))
    batch = [ds[i] for i in range(batch_size)]
    
    print(f"Batch size: {batch_size}")
    for i, mol in enumerate(batch):
        print(f"Molecule {i}: {len(mol['atom_labels'])} atoms, labels: {mol['atom_labels']}")
    print()
    
    # Test collate function
    tokenizer = ChemBERTaWithCoords.get_tokenizer()
    
    print("=== Individual SMILES Generation ===")
    for i, mol in enumerate(batch):
        smiles = mol_to_explicit_smiles(mol['bonds'], mol['atom_labels'])
        print(f"Molecule {i}: {smiles}")
    print()
    
    print("=== Collate Function Output ===")
    collated = collate(batch, tokenizer)
    
    print(f"Collated batch keys: {list(collated.keys())}")
    for key, value in collated.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)} (length: {len(value)})")
    
    # Detailed inspection
    print(f"\nInput IDs:\n{collated['input_ids']}")
    print(f"\nCoordinates shape: {collated['coords'].shape}")
    print(f"MM reduced shape: {collated['mm_reduced'].shape}")
    print(f"Mask shape: {collated['mask'].shape}")
    print(f"Mask content:\n{collated['mask']}")
    
    print(f"\nToken2atom mappings:")
    for i, mapping in enumerate(collated['token2atom']):
        print(f"  Molecule {i}: {mapping}")
    print()
    
    return collated

def test_model_forward():
    """Test the model forward pass with real data."""
    print("=== Testing Model Forward Pass ===\n")
    
    # Get collated batch
    ds = OCNMoleculeDataset(
        root='./raw',
        filenames=['5v.csv'],
        processed_dir='./processed',
        augmentations=None,
    )
    
    batch = [ds[0]]  # Single molecule
    tokenizer = ChemBERTaWithCoords.get_tokenizer()
    collated = collate(batch, tokenizer)
    
    print("=== Model Creation ===")
    model = ChemBERTaWithCoords(pretrained_name='seyonec/ChemBERTa-zinc-base-v1', lora_r=8)
    model.eval()  # Set to evaluation mode
    
    print("=== Forward Pass ===")
    with torch.no_grad():
        mm_pred = model(
            input_ids=collated['input_ids'],
            attention_mask=collated['attention_mask'],
            coords=collated['coords'],
            token2atom=collated['token2atom'],
            mask=collated['mask'],
        )
    
    print(f"Prediction shape: {mm_pred.shape}")
    print(f"Predictions: {mm_pred}")
    print(f"Target shape: {collated['mm_reduced'].shape}")
    print(f"Targets: {collated['mm_reduced']}")
    
    # Check if dimensions match
    if mm_pred.shape == collated['mm_reduced'].shape:
        print("✅ Prediction and target shapes match!")
    else:
        print("❌ Shape mismatch between predictions and targets!")
    
    # Calculate sample loss
    mask = collated['mask'].bool()  # Convert to boolean for indexing
    if mask.sum() > 0:
        loss = torch.nn.functional.mse_loss(mm_pred[mask], collated['mm_reduced'][mask])
        print(f"Sample loss: {loss.item():.6f}")
    
    print()

def main():
    """Run all tests."""
    print("🧪 Starting Data Flow Tests for MAG-OCN\n")
    
    try:
        # Test 1: Single molecule
        mol, smiles, token2atom = test_single_molecule()
        
        # Test 2: Batch collate
        collated = test_batch_collate()
        
        # Test 3: Model forward pass
        test_model_forward()
        
        print("🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
