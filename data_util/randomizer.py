# randomizer.py

import torch
import numpy as np
from typing import Optional

from .graph_utils import _build_adjacency, _all_pairs_shortest_paths

__all__ = ['OCNRandomRotation', 'OCNRandomReflection', 'OCNRandomMicroPerturbation']

class OCNRandomRotation:
    """Randomly rotate a molecule in 3D space.

    Rotation matrices are sampled uniformly from SO(3) by composing three
    uniform angles.  Distances—and therefore *adjacency* and *SPD*—stay intact,
    so no graph recomputation is required.
    """
    def __call__(self, mol: dict) -> dict:
        coords = mol['coords']  # torch view, no copy

        # generate Euler angles
        ax, ay, az = torch.rand(3) * 2 * np.pi
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(ax), -torch.sin(ax)],
            [0, torch.sin(ax),  torch.cos(ax)],
        ])
        Ry = torch.tensor([
            [ torch.cos(ay), 0, torch.sin(ay)],
            [0, 1, 0],
            [-torch.sin(ay), 0, torch.cos(ay)],
        ])
        Rz = torch.tensor([
            [torch.cos(az), -torch.sin(az), 0],
            [torch.sin(az),  torch.cos(az), 0],
            [0, 0, 1],
        ])

        R = Rx @ Ry @ Rz  # [3,3]
        mol["coords"] = coords @ R.T
        return mol
    
class OCNRandomReflection:
    """Mirror the molecule across a random coordinate plane (X, Y or Z)."""

    def __call__(self, mol: dict) -> dict:
        axis = torch.randint(0, 3, (1,)).item()  # 0,1,2
        mol['coords'][:, axis] *= -1  # flip sign along chosen axis
        return mol
    
class OCNRandomMicroPerturbation:
    """Apply small Gaussian noise to coordinates **and** magnetic moments.

    Because coordinate noise breaks the original bond distances, we
    *recompute* the adjacency & SPD matrices so they remain consistent.

    Parameters
    ----------
    position_noise : float, default 0.01 Å
        Standard deviation of the isotropic Gaussian noise added to each atom
        coordinate.
    moment_noise : float, default 0.03 (log-scaled units)
        Std-dev of the Gaussian noise added to the *reduced* magnetic moment
        targets (``mol['targets']``).
    thresh : float, default 2.0 Å
        Distance cut-off passed to the rule-based bonding function.
    """

    def __init__(self, position_noise: float = 0.01, moment_noise: float = 0.03, threshold: float = 2.0):
        self.position_noise = position_noise
        self.moment_noise = moment_noise
        self.threshold = threshold

    def __call__(self, mol: dict) -> dict:
        # coordinate jitter
        coords = mol["coords"]
        mol["coords"] = coords + torch.randn_like(coords) * self.position_noise

        # recompute adjacency & SPD (CPU numpy)
        new_coords = mol["coords"].cpu().numpy()
        labels = mol["atom_labels"]
        adj = _build_adjacency(new_coords, labels, self.threshold)
        spd = _all_pairs_shortest_paths(adj)
        mol['bonds'] = torch.from_numpy(adj).float()
        mol['spd'] = torch.from_numpy(spd).float()

        # magnetic-moment jitter
        if self.moment_noise > 0:
            noise = torch.randn_like(mol["mm_reduced"]) * self.moment_noise
            candidate = mol["mm_reduced"] + noise
            mol["mm_reduced"] = torch.sign(mol["mm_reduced"]) * torch.abs(candidate)

        return mol