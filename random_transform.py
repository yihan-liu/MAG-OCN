
import torch
import numpy as np
import pandas as pd


def rotate_points(points_df):
    """
    Rotates a set of 3D points around a random axis through the origin.

    Parameters:
    points_df (pd.DataFrame): DataFrame with columns ['x', 'y', 'z'] representing 3D points.

    Returns:
    pd.DataFrame: DataFrame with rotated points.
    """
    # Normalize the random axis vector
    axis_vector = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
    axis_vector = axis_vector / np.linalg.norm(axis_vector)

    # Form a random angle
    angle = np.random.rand() * 2 * np.pi

    # Components of the axis vector
    ux, uy, uz = axis_vector

    # Compute the rotation matrix using Rodrigues' rotation formula
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    one_minus_cos = 1 - cos_theta

    rotation_matrix = np.array([
        [
            cos_theta + ux**2 * one_minus_cos,
            ux * uy * one_minus_cos - uz * sin_theta,
            ux * uz * one_minus_cos + uy * sin_theta
        ],
        [
            uy * ux * one_minus_cos + uz * sin_theta,
            cos_theta + uy**2 * one_minus_cos,
            uy * uz * one_minus_cos - ux * sin_theta
        ],
        [
            uz * ux * one_minus_cos - uy * sin_theta,
            uz * uy * one_minus_cos + ux * sin_theta,
            cos_theta + uz**2 * one_minus_cos
        ]
    ])

    # Extract points as a numpy array
    points_array = points_df[['X', 'Y', 'Z']].to_numpy()

    # Rotate points
    rotated_points = points_array @ rotation_matrix.T

    # Create a DataFrame with rotated points
    rotated_df = pd.DataFrame(rotated_points, columns=['X', 'Y', 'Z'])

    return rotated_df

def translate_points(points_df, region = 10.0):
    """
    Translates a set of 3D points.

    Parameters:
    points_df (pd.DataFrame): DataFrame with columns ['x', 'y', 'z'] representing 3D points.
    region: the maximized distance each point can move. 

    Returns:
    pd.DataFrame: DataFrame with translated points.
    """
    # Normalize the random axis vector
    axis_vector = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
    axis_vector = axis_vector / np.linalg.norm(axis_vector)

    # Times the axis vector with the distance moved
    translation_vector = axis_vector * np.random.rand() * region

    # Add the translation vector to each point
    translated_points = points_df[['X', 'Y', 'Z']].to_numpy() + translation_vector

    # Create a DataFrame with translated points
    translated_df = pd.DataFrame(translated_points, columns=['X', 'Y', 'Z'])

    return translated_df

def mirror_points(points_df):
    """
    Mirrors (reflects) a set of 3D points.

    Parameters:
    points_df (pd.DataFrame): DataFrame with columns ['x', 'y', 'z'] representing 3D points.

    Returns:
    pd.DataFrame: DataFrame with mirrored points.
    """
    mirrored_points = points_df.copy()
    if (np.random.rand() > 0.5):
        mirrored_points['X'] = -mirrored_points['X']

    if (np.random.rand() > 0.5):
        mirrored_points['Y'] = -mirrored_points['Y']
    
    if (np.random.rand() > 0.5):
        mirrored_points['Z'] = -mirrored_points['Z']
    
    return mirrored_points

def transform_points(points_df):
    points = rotate_points(points_df)
    points = translate_points(points)
    points = mirror_points(points)
    return points

class RandomBox:
    """
    Draws a random 3D box in th eoriginal coordinate space such
    """
