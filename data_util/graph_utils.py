# graph_utils.py
# Utility functions for building molecular graphs and computing distances

import numpy as np
from collections import deque

def _dist(a1, a2):
    """Compute Euclidean distance between two points."""
    return np.linalg.norm(a1 - a2)

def _build_adjacency(coords, labels, threshold=2.0):
    """Rule-based connectivity for C/N/O atoms.
    
    * Carbon: up to three nearest C/N/O ≤ ``thresh`` Å.
    * Nitrogen: up to three nearest **carbon** atoms ≤ ``thresh`` Å.
    * Oxygen: no outgoing rules (can still be targeted by carbon).
    """
    n = len(coords)
    adj = np.zeros((n, n), dtype=float)

    for i in range(n):
        dists = [(_dist(coords[i], coords[j]), j) for j in range(n) if j != i]
        dists.sort(key=lambda x: x[0])
        bonded = 0

        if labels[i] == 'C':
            # C forms bonds with up to three nearest atoms (C, N, or O) within threshold.
            for d, j in dists:
                if bonded >= 3:
                    break
                if d <= threshold and labels[j] in {'C', 'N', 'O'}:
                    adj[i, j] = 1
                    adj[j, i] = 1
                    bonded += 1
        elif labels[i] == 'N':
            # N forms bonds with up to three nearest carbon atoms within threshold.
            for d, j in dists:
                if bonded >= 3:
                    break
                if d <= threshold and labels[j] == 'C':
                    adj[i, j] = 1
                    adj[j, i] = 1
                    bonded += 1
        # For oxygen atoms, no explicit bonding rule is defined
    return adj

def _all_pairs_shortest_paths(adj):
    """Breadth-first all-pairs shortest paths for *small* graphs (≤100 nodes).

    Returns a float matrix where ``spd[i, j]`` is the *hop-count* (0-∞).
    Unreachable pairs are set to ``1e9`` so they can be masked later.
    """
    n = adj.shape[0]
    spd = np.full((n, n), 1e9, dtype=float)
    for src in range(n):
        spd[src, src] = 0
        q = deque([src])
        while q:
            v = q.popleft()
            for nb in np.where(adj[v] > 0)[0]:
                if spd[src, nb] > spd[src, v] + 1:
                    spd[src, nb] = spd[src, v] + 1
                    q.append(nb)

    return spd
