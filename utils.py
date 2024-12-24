
import numpy as np
import torch

ATOM_TYPE_DICT = {
    'N': 0,
    'C': 1,
    'H': 2,
    'O': 3,
}


def atom_type_to_onehot(atom_label: str):
    """
    Convert e.g. 'N1' -> 'N' -> one-hot vector [1,0,0,0]
    """
    element = ''.join([ch for ch in atom_label if ch.isalpha])
    one_hot = [0] * len(ATOM_TYPE_DICT)
    idx = ATOM_TYPE_DICT[element]
    one_hot[idx] = 1
    return one_hot

def pairwise_edges(positions, cutoff=2.0):
    """
    Build an edge index for all pairs of atoms within 'cutoff' distance.
    Inputs positions: (num_nodes, 3)
    Returns edge_index: (2, E)
    """
    num_nodes = positions.shape[0]
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= cutoff:
                edges.append([i, j])
                edges.append([j, i])
    if len(edges) == 0:
        # NOTE: no chemical bonds are formed
        edges = [[0, 0]]
    return torch.tensor(edges, dtype=torch.long).t()  # Shape (2, E)