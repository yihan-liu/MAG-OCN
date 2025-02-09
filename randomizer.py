# randomizer.py

import copy
import numpy as np

class OCNRandomTranslation:
    """
    Randomly translates all atom positions by a vector within a specified range.
    """
    def __init__(self, max_translation=2.0):
        self.max_translation = max_translation

    def __call__(self, molecule):
        # Generate a random translation vector
        translation = np.random.uniform(-self.max_translation, self.max_translation, size=(1, 3))
        molecule['features'][:, 3:6] += translation
        return molecule
    
class OCNRandomRotation:
    """
    Randomly rotates the molecule in 3D space.
    """
    def __init__(self):
        pass

    def __call__(self, molecule):
        coords = molecule['features'][:, 3:6]

        # Generate the angles
        angle_x = np.random.rand() * 2 * np.pi
        angle_y = np.random.rand() * 2 * np.pi
        angle_z = np.random.rand() * 2 * np.pi

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angle_x), -np.sin(angle_x)],
                       [0, np.sin(angle_x), np.cos(angle_x)]])
        
        Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                       [0, 1, 0],
                       [-np.sin(angle_y), 0, np.cos(angle_y)]])
        
        Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                       [np.sin(angle_z), np.cos(angle_z), 0],
                       [0, 0, 1]])
        
        R = Rx @ Ry @ Rz  # full 3D rotation matrix

        coords_rotated = coords @ R.T
        molecule['features'][:, 3:6] = coords_rotated
        return molecule
    
class OCNRandomReflection:
    """
    Randomly mirrors the molecule across a plane
    """
    def __init__(self):
        pass

    def __call__(self, molecule):
        coords = molecule['features'][:, 3:6]

        # Generate the reflection plane
        axis = np.random.randint(0, 3)  # 0=X, 1=Y, 2=Z
        coords[:, axis] *= -1  # flip the chosen coord axis

        molecule['features'][:, 3:6] = coords
        return molecule
    
class OCNRandomMicroPerturbation:
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

    def __call__(self, molecule):
        # Add noise to coordinates
        coords = molecule['features'][:, 3:6]
        noise_coords = np.random.randn(*coords.shape) * self.position_noise
        molecule['features'][:, 3:6] = coords + noise_coords

        # Add noise to magnetic moments
        noise_mm = np.random.randn(*molecule['targets'].shape) * self.moment_noise
        candidate_mm = molecule['targets'] + noise_mm

        sign_original = np.sign(molecule['targets'])
        molecule['targets'] = sign_original * np.abs(candidate_mm)
        return molecule