# preprocessor.py

import os
import copy
import argparse
import hashlib
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import deque

from .atoms_encoding import ATOM_DICT
from .graph_utils import _build_adjacency, _all_pairs_shortest_paths
from .randomizer import *

class OCNMoleculeDataset(Dataset):
    """Variable-length molecule dataset with cached preprocessing.

    Each *molecule* is stored as a dictionary of tensors.

    Extra feature: an **all-pairs shortest-path distance (SPD) matrix** for
    graph-aware models.  Unreachable nodes are given a large sentinel value
    (``1e9``) so they can be masked later.

    Returned sample dict
    --------------------
    ``{
        'atom_labels'  : list[str]          (chemical symbol per atom)
        'bonds'        : FloatTensor [N, N]  (0/1 bond matrix)
        'coords'       : FloatTensor [N, 3]  (xyz coordinates in Å, centered)
        'mm_reduced'   : FloatTensor [N]     (reduced magnetic moment per atom)
        'mm_original'  : FloatTensor [N]     (original MM, *not* reduced)
        'v_value'      : int                (penta-ring count from filename)
    }``
    where ``N`` is the number of atoms in that molecule (variable).
    """

    def __init__(
        self,
        root: str,
        filenames: list[str],
        threshold: float = 2.0,
        augmentations: list[callable] = None,
        processed_dir: str = './processed',
        seed: int = 42
    ):
        super().__init__()

        self.root = root
        if isinstance(filenames, str):
            self.filenames = [filenames]
        else:
            self.filenames = filenames
        self.threshold = threshold
        self.augmentations = augmentations
        self.processed_dir = processed_dir

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # cache filepath
        folder_name = '-'.join([fn.replace('.csv', '') for fn in self.filenames])
        dataset_dir = os.path.join(self.processed_dir, folder_name)

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        # Create a unique hash for the parameters based on user-specified order
        folder_name = "-".join([fn.replace(".csv", "") for fn in self.filenames])
        os.makedirs(os.path.join(processed_dir, folder_name), exist_ok=True)
        params_hash = hashlib.md5(
            f"{''.join(self.filenames)}_{threshold}_{seed}".encode()
        ).hexdigest()
        cache_fp = os.path.join(processed_dir, folder_name, f"{params_hash}.pt")

        if os.path.exists(cache_fp):
            print(f"[OCN] Loading processed molecules from {cache_fp}")
            self.molecules = torch.load(cache_fp)
        else:
            print(f"[OCN] Processing CSV → tensors (cache: {cache_fp})")
            self.molecules = []
            for fn in self.filenames:
                mol = self._process_csv(os.path.join(root, fn))
                if mol is not None:
                    self.molecules.append(mol)
            torch.save(self.molecules, cache_fp)

    def __len__(self):
        return len(self.molecules) 
    
    def __getitem__(self, idx):
        mol = copy.deepcopy(self.molecules[idx])

        # on‑the‑fly augmentations (coords + mm noise, etc.)
        if self.augmentations:
            if isinstance(self.augmentations, list):
                for aug in self.augmentations:
                    mol = aug(mol)
            else:
                mol = self.augmentations(mol)

        return mol

    def _process_csv(self, filepath):
        """Convert a single CSV into a dictionary of tensors.
        
        Returns ``None`` if the file is unreadable or contains no valid atoms.
        """
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f'[OCN] Error reading {filepath}: {e}')
            return None

        # v-value from filename
        try:
            v_value = int(os.path.basename(filepath).split('_')[0])
        except ValueError:
            v_value = 0

        atom_labels, coords, mm_original, mm_reduced = [], [], [], []
        for _, row in df.iterrows():
            atom = row['ATOM'][0]
            if atom not in ATOM_DICT:
                continue  # Skip atoms not in our vocabulary (e.g., H)
            
            atom_labels.append(atom)
            coords.append([row['X'], row['Y'], row['Z']])
            mm = row['MAGNETIC_MOMENT']
            mm_original.append(mm)
            mm_reduced.append(self._reduce_mm(mm))
        
        if len(coords) == 0:
            print(f'[OCN] No valid atoms in {filepath}')
            return None
        
        coords = np.array(coords, dtype=np.float32)
        coords -= coords.mean(axis=0, keepdims=True)  # center coordinates

        adj = _build_adjacency(coords, atom_labels, threshold=self.threshold)

        return {
            'atom_labels': atom_labels,
            'bonds': torch.tensor(adj, dtype=torch.float32),
            'coords': torch.tensor(coords, dtype=torch.float32),
            'mm_reduced': torch.tensor(mm_reduced, dtype=torch.float32),
            'mm_original': torch.tensor(mm_original, dtype=torch.float32),
            'v_value': v_value,
        }

    @staticmethod
    def _reduce_mm(y):
        '''
        Normalize the magnetic moment using: sign(y) * log(1 + |y|)
        This helps with training stability by reducing the range of target values.
        '''
        return np.sign(y) * np.log1p(np.abs(y))
    
    @staticmethod
    def _recover_mm(y):
        '''
        Recover original magnetic moment using: sign(y) * (exp(|y'|) - 1)
        Use this to convert model predictions back to original scale.
        '''
        return np.sign(y) * np.expm1(np.abs(y))