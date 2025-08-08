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

def token2atom_mapping(smiles: str, tokenizer, n_atoms: int) -> List[int]:
    """Map tokens to atoms in the SMILES string.
    
    This is a simplified mapping that assumes token order roughly matches atom order.
    For more accurate mapping, we'd need a proper SMILES parser.
    """
    tokens = tokenizer.tokenize(smiles)
    
    # Simple heuristic: map tokens to atoms cyclically, skipping special tokens
    mapping = []
    atom_idx = 0
    
    for i, token in enumerate(tokens):
        # Skip special tokens like [CLS], [SEP], [PAD]
        if token.startswith('[') and token.endswith(']'):
            mapping.append(0)  # Default to first atom for special tokens
        else:
            mapping.append(atom_idx % n_atoms)
            atom_idx += 1
    
    # Add mappings for [CLS] and [SEP] tokens that tokenizer adds
    full_mapping = [0] + mapping + [0]  # [CLS] + tokens + [SEP]
    
    return full_mapping[:len(tokens) + 2]  # Ensure correct length

def collate(batch: List[dict], tokenizer) -> dict:
    """Custom collate function to handle variable-size molecules."""
    B = len(batch)
    N_max = max(mol['coords'].size(0) for mol in batch)

    coords_pad = torch.zeros(B, N_max, 3)         # [B, N_max, 3]
    mm_reduced_pad = torch.zeros(B, N_max)           # [B, N_max]
    mask_pad = torch.zeros(B, N_max, dtype=torch.bool)  # [B, N_max]

    smiles_list, token2atom_list = [], []
    for b, mol in enumerate(batch):
        N = mol['coords'].size(0)
        coords_pad[b, :N] = mol['coords']
        mm_reduced_pad[b, :N] = mol['mm_reduced']
        mask_pad[b, :N] = 1

        smiles = mol_to_explicit_smiles(mol['bonds'], mol['atom_labels'])
        smiles_list.append(smiles)
        token2atom_list.append(token2atom_mapping(smiles, tokenizer, N))
    
    tok = tokenizer(smiles_list, return_tensors='pt', padding=True)
    return {
        'input_ids': tok['input_ids'],
        'attention_mask': tok['attention_mask'],

        'coords': coords_pad,
        'mm_reduced': mm_reduced_pad,
        'mask': mask_pad,
        'token2atom': token2atom_list,
    }