
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
    
class RandomMicroPerturbation:
    """
    A transform that applies micro-level perturbations to atom positions (x, y, z)
    and the target magnetic moment.
    
    Args:
        position_noise (float): Standard deviation of Gaussian noise for positions.
        moment_noise (float): Standard deviation of Gaussian noise for magnetic moments.
    """
    def __init__(self, position_noise=0.01, moment_noise=0.01):
        self.position_noise = position_noise
        self.moment_noise = moment_noise

    def __call__(self, data: Data):
        # Store the original magnetic moments
        if not hasattr(data, 'original_y'):
            data.original_y = data.y.clone()

        coords = data.x[:, -3:]

        # Add noise to coordinates
        noise_coords = torch.randn_like(coords) * self.position_noise
        data.x[:, -3:] = coords + noise_coords

        # Add noise to magnetic moments
        # data.y typically has shape: (num_atoms,) or (num_atoms, 1)
        noise_moment = torch.randn_like(data.y) * self.moment_noise
        data.y = data.y + noise_moment
        return data
    
    @staticmethod
    def get_original_y(data):
        return getattr(data, 'original_y', None)

def macro_randomize(data: Data, max_translation=0.5, macro_transforms=None):
    """
    Applying random transformations to the original molecule.

    Args:
        original_data (Data): The atom coords of the original molecule, Defaults (None) lead to all transformations.
        max_translation (float, optional): Maximum displacement of the molecule, Defaults to 0.5.
        macro_transforms (list, optional): List of macro-transformations objects (RandomTranslation, RandomRotation, and RandomReflection). Defaults to None.

    Returns:
        Data: Augmented Data object
    """
    randomized_data = data.clone()

    if macro_transforms == None:
        # default behavior
        macro_transforms = [
            RandomTranslation(max_translation=max_translation),
            RandomRotation(),
            RandomReflection()
        ]

    # Perform each transform
    for transform in macro_transforms:
        randomized_data = transform(randomized_data)
    return randomized_data


def micro_randomize(data: Data, position_noise=0.01, moment_noise=0.01):
    """
    A transform that applies micro-level perturbations to atom positions and the target magnetic moment

    Args:
        data (Data): The atom coords of the original molecule
        position_noise (float, optional): Standard deviation of Gaussian noise for position perturbation. Defaults to 0.01.
        moment_noise (float, optional): Standard deviation of Gaussian noise for moment perturbation. Defaults to 0.01.

    Returns:
        Data: Perturbed Data object
    """
    randomized_data = data.clone()

    # perform micro-transform
    transform = RandomMicroPerturbation(position_noise, moment_noise)
    randomized_data = transform(randomized_data)
    
    # record the unaffected magnetic moment
    original_y = transform.get_original_y(randomized_data)
    return randomized_data, original_y

def augment_data(
        data: Data,
        num_new_data=100,
        max_translation=0.5,
        macro_transforms=None,
        position_noise=0.01,
        moment_noise=0.01,
        num_micro_per_macro=1
):
    """
    Generate data that:
        1) Applies macro_randomize to get macro-level transforms
        2) Applies micro_randomize to get multiple 'fine-grained' variations

    Args:
        data (Data): Original single-molecule data.
        num_new_data (int, optional): Number of augmented samples. Defaults to 100.
        max_translation (float, optional): Amplitude of random translation of the molecule. Defaults to 0.5.
        macro_transforms (list, optional): List of macro-transforms to perform. Defaults to None.
        position_noise (float, optional): Std for position noise in micro_randomize. Defaults to 0.01.
        moment_noise (float, optional): Std for moment noise in micro_randomize. Defaults to 0.01.
        num_micro_per_macro (int, optional): For each sample, the number of micro-perturbations to perform for each macro-transform. Defaults to 1.

    Returns:
        list[Data]: A list of augmented 'Data' Objects.
        list[int]: A list of all original magnetic moments.
    """
    randomized_datasets = []
    original_ys = []

    for _ in range(num_new_data):
        new_data = data.clone()
        new_data = macro_randomize(new_data, max_translation, macro_transforms)

        for _ in range(num_micro_per_macro):
            #  Perform perturbations
            perturbed_data, original_y = micro_randomize(new_data, position_noise, moment_noise)
            randomized_datasets.append(perturbed_data)
            original_ys.append(original_y)

    return randomized_datasets, original_ys