
import torch
from torch_geometric.data import Data
import numpy as np

class RandomTranslation:
    """
    Randomly translates all atom positions by a vector within a specified range.
    """
    def __init__(self, max_translation=2.0):
        self.max_translation = max_translation

    def __call__(self, data):
        # Generate a random translation vector
        translation = torch.FloatTensor(3).uniform_(-self.max_translation, self.max_translation)
        data.x[:, -3:] += translation
        return data
    
class RandomRotation:
    """
    Randomly rotates the molecule in 3D space.
    """
    def __init__(self):
        pass

    def __call__(self, data):
        coords = data.x[:, -3:]

        # Generate the angles
        angle_x = torch.rand(1).item() * 2 * np.pi
        angle_y = torch.rand(1).item() * 2 * np.pi
        angle_z = torch.rand(1).item() * 2 * np.pi

        Rx = torch.tensor([[1, 0, 0],
                           [0, np.cos(angle_x), -np.sin(angle_x)],
                           [0, np.sin(angle_x), np.cos(angle_x)]], dtype=torch.float)
        
        Ry = torch.tensor([[np.cos(angle_y), 0, np.sin(angle_y)],
                           [0, 1, 0],
                           [-np.sin(angle_y), 0, np.cos(angle_y)]], dtype=torch.float)
        
        Rz = torch.tensor([[np.cos(angle_z), -np.sin(angle_z), 0],
                           [np.sin(angle_z), np.cos(angle_z), 0],
                           [0, 0, 1]], dtype=torch.float)
        
        R = Rx @ Ry @ Rz  # full 3D rotation matrix

        coords_rotated = coords @ R.T
        data.x[:, -3:] = coords_rotated
        return data
    
class RandomReflection:
    """
    Randomly mirrors the molecule across a plane
    """
    def __init__(self):
        pass

    def __call__(self, data):
        coords = data.x[:, -3:]

        # Generate the reflection plane
        axis = torch.randint(0, 3, (1,)).item()  # 0=X, 1=Y, 2=Z
        coords[:, axis] *= -1  # flip the chosen coord axis

        data.x[:, -3:] = coords
        return data

def expand_dataset(original_data: Data, num_samples=100, transforms=None):
    """
    Expands the dataset by applying random transformations to the original molecule.

    Args:
        original_data (Data): The atom coords of the original molecule
        num_samples (int, optional): Number of generated samples. Defaults to 100.
        transforms (list, optional): List of transformations objects (RandomTranslation, RandomRotation, and RandomReflection). Defaults to None.

    Returns:
        list[Data]: A list of augmented Data objects
    """
    augmented_data = []
    for _ in range(num_samples):
        new_data = original_data.clone()
        for transform in transforms:
            new_data = transform(new_data)
        augmented_data.append(new_data)
    return augmented_data