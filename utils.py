# utils.py

from typing import List
import torch
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops

def mol_to_explicit_smiles(adj, atom_labels) -> str:
    """Convert bond matrix and atom labels to an explicit SMILES string."""
    mol = Chem.RWMol()
    atom_ids = []

    for sym in atom_labels:
        atom = Chem.Atom(sym)
        atom.SetNumExplicitHs(0)  # no implicit hydrogens
        atom_ids.append(mol.AddAtom(atom))

    for i, _ in enumerate(atom_labels):
        for j in range(i + 1, len(atom_labels)):
            if adj[i, j] > 0:
                mol.AddBond(i, j, Chem.BondType.SINGLE)
    
    mol = mol.GetMol()
    mol.UpdatePropertyCache(strict=False)  # Calculate implicit valence
    mol = rdmolops.AddHs(mol, addCoords=False)  # add implicit hydrogens
    return rdmolfiles.MolToSmiles(mol, isomericSmiles=True)

def token2atom_mapping(smiles: str, tokenizer, n_atoms: int, max_length: int = 512) -> List[int]:
    """Map tokens to atoms in the SMILES string, handling spatial segments better.
    
    For spatially segmented molecules, we have fewer atoms per segment so truncation
    is less likely to be an issue. This mapping assumes token order roughly matches
    atom order within the spatial segment.
    
    Args:
        smiles: SMILES string for the spatial segment
        tokenizer: HuggingFace tokenizer
        n_atoms: Number of atoms in this spatial segment
        max_length: Maximum sequence length (including special tokens)
    """
    tokens = tokenizer.tokenize(smiles)
    
    # Account for [CLS] and [SEP] tokens in max_length
    max_content_tokens = max_length - 2
    
    # For spatial segments, we should rarely need truncation since segments are smaller
    if len(tokens) > max_content_tokens:
        tokens = tokens[:max_content_tokens]
        print(f"Warning: Truncating segment SMILES from {len(tokenizer.tokenize(smiles))} to {len(tokens)} tokens")
    
    # Improved mapping: distribute tokens more evenly across atoms
    mapping = []
    
    if len(tokens) == 0:
        mapping = [0]  # Fallback for empty tokenization
    else:
        # Map tokens to atoms more evenly using interpolation
        for i, token in enumerate(tokens):
            # Skip special tokens like [CLS], [SEP], [PAD]
            if token.startswith('[') and token.endswith(']'):
                mapping.append(0)  # Default to first atom for special tokens
            else:
                # Map token index to atom index proportionally
                atom_idx = int((i / len(tokens)) * n_atoms) % n_atoms
                mapping.append(atom_idx)
    
    # Add mappings for [CLS] and [SEP] tokens that tokenizer adds
    full_mapping = [0] + mapping + [0]  # [CLS] + tokens + [SEP]
    
    return full_mapping

def collate(batch: List[dict], tokenizer, max_length: int = 512) -> dict:
    """Custom collate function for spatial segments with optimized padding."""
    B = len(batch)
    N_max = max(mol['coords'].size(0) for mol in batch)

    coords_pad = torch.zeros(B, N_max, 3)         # [B, N_max, 3]
    mm_reduced_pad = torch.zeros(B, N_max)        # [B, N_max]
    mask_pad = torch.zeros(B, N_max, dtype=torch.bool)  # [B, N_max]

    smiles_list, token2atom_list = [], []
    segment_ids, original_indices_list = [], []
    
    for b, mol in enumerate(batch):
        N = mol['coords'].size(0)
        coords_pad[b, :N] = mol['coords']
        mm_reduced_pad[b, :N] = mol['mm_reduced']
        mask_pad[b, :N] = 1

        smiles = mol_to_explicit_smiles(mol['bonds'], mol['atom_labels'])
        smiles_list.append(smiles)
        
        # Generate improved token2atom mapping for spatial segments
        token2atom = token2atom_mapping(smiles, tokenizer, N, max_length)
        token2atom_list.append(token2atom)
        
        # Store segment metadata if available
        if 'segment_id' in mol:
            segment_ids.append(mol['segment_id'])
        if 'original_indices' in mol:
            original_indices_list.append(mol['original_indices'])
    
    # Tokenize with truncation to handle long sequences
    tok = tokenizer(
        smiles_list, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=max_length
    )
    
    result = {
        'input_ids': tok['input_ids'],
        'attention_mask': tok['attention_mask'],
        'coords': coords_pad,
        'mm_reduced': mm_reduced_pad,
        'mask': mask_pad,
        'token2atom': token2atom_list,
    }
    
    # Add segment metadata to batch if available
    if segment_ids:
        result['segment_ids'] = torch.tensor(segment_ids, dtype=torch.long)
    if original_indices_list:
        result['original_indices'] = original_indices_list
    
    return result