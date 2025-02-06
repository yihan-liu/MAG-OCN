# preprocessor.py

import os

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
import pandas as pd


ATOM_TYPE_DICT = {
    'N': 0,
    'C': 1,
    'O': 2,
}

def atom_encode(element: str):
    """
    Convert e.g. 'N1' -> 'N' -> one-hot vector [1,0,0,0]
    """
    if element not in ATOM_TYPE_DICT:
        return None

    one_hot = [0] * len(ATOM_TYPE_DICT)
    idx = ATOM_TYPE_DICT[element]
    one_hot[idx] = 1
    return one_hot

class AtomDataset(InMemoryDataset):
    def __init__(self, root, filename,
                 transform=None, pre_transform=None, threshold=2.0):
        """
        root: path containing the 'atoms.csv'
        threshold: distance threshold for constructing edges
        """
        self.threshold = threshold
        self.filename = filename
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # return the raw file name
        return [self.filename]
    
    @property
    def processed_file_names(self):
        # return the processed file name
        return [os.path.splitext(self.filename)[0] + '.pt']
    
    @property
    def raw_dir(self):
        # always look for raw data in the 'raw' folder
        return os.path.join(self.root, 'raw')
    
    @property
    def processed_dir(self):
        # override to save the processed file to the specified
        return os.path.join(self.root, 'processed')
    
    def process(self):
        df = pd.read_csv(self.raw_paths[0])

        # Lists for nodes
        features = []   # each element: one_hot + [x, y, z]
        labels = []     # store atom type letter: 'N', 'C' or 'O'
        mags = []        # corresponding magnetic moment

        # process each row of csv file
        for _, row in df.iterrows():
            atom_label = row['ATOM']  # e.g. 'N1'
            atom_type = atom_label[0]
            one_hot = atom_encode(atom_type)  # encode the element
            if one_hot is None:
                continue  # skip atoms not in the dict
            labels.append(atom_label[0])
            # append coordinates
            coords = [row['X'], row['Y'], row['Z']]
            features.append(one_hot + coords)
            labels.append(atom_type)
            mags.append(row['MAGNETIC_MOMENT'])

        # Normalize atom locations (only use translation)
        features = np.array(features)
        coords = features[:, -3:]
        center = np.mean(coords, axis=0)
        features[:, -3:] = coords - center

        # Combine measured nodes with placeholders
        self.node_features = features
        self.atom_labels = labels

        # Process MM
        # for measured nodes, apply the transformation: y = sign(y) * log(1 + |y|)
        mags_tensor = torch.tensor(mags, dtype=torch.float).view(-1, 1)
        transformed_mags = torch.sign(mags_tensor) * torch.log1p(torch.abs(mags_tensor))

        # create the tensor for node features
        x = torch.tensor(self.node_features, dtype=torch.float)  # shape (num_nodes, 7)
        
        # Construct edge_index using only measured nodes
        self._generate_edges()

        # Construct Data
        data = Data(x=x, edge_index=self.edge_index, y=transformed_mags)

        data_list = [data]
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])

    def _generate_edges(self):
        """
        Build an edge index among the measured (non-placeholder) atoms based on the following rules.
        - Carbon atoms (C) form bonds with up to three nearest atoms (C, N, O) within the threshold.
        - Nitrogen atoms (N) form bonds with up to three nearest carbon atoms within the threshold.
        """
        positions = self.node_features[:self.measured_count, -3:]
        num_nodes = positions.shape[0]
        edges = []

        for i in range(num_nodes):
            current_label = self.atom_labels[i]
            distances = []

            # Compute distances to all other atoms
            for j in range(num_nodes):
                if i == j:
                    continue
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append((dist, j))

            # Sort neighbors by distance
            distances.sort(key=lambda x: x[0])
            
            if current_label == 'C':
                # C forms bonds with up to 3 nearest atoms (C, N, O)
                bonded_atoms = 0
                for dist, j in distances:
                    if bonded_atoms >= 3:
                        break
                    if dist <= self.threshold and self.atom_labels[j] in {'C', 'N', 'O'}:
                        edges.append([i, j])
                        edges.append([j, i])
                        bonded_atoms += 1

            elif current_label == 'N':
                # N forms bonds with up to 3 nearest carbon atoms
                bonded_atoms = 0
                for dist, j in distances:
                    if bonded_atoms >= 3:
                        break
                    if dist <= self.threshold and self.atom_labels[j] == {'C'}:
                        edges.append([i, j])
                        edges.append([j, i])
                        bonded_atoms += 1
                        
        if len(edges) == 0:
            # No bonds are formed
            edges = [[0, 0]]

        self.edge_index = torch.tensor(edges, dtype=torch.long).t()  # Shape (2, E)

if __name__ == '__main__':
    dataset = AtomDataset(root='./', filename='0v.csv', threshold=2.0)