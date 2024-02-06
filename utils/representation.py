import jax.numpy as jnp
from jraph import GraphsTuple
from pymatgen.core import Structure
import numpy as np
from typing import Dict, List, Optional, Union, Callable

def get_coordinates(structure:Structure):
    atomic_numbers = np.array([x.specie.number for x in structure.sites])
    coordinates = np.array([x.coords for x in structure.sites])
    n_sites = structure.num_sites
    return atomic_numbers, coordinates, n_sites

def get_coulomb_matrix(structure:Structure):
    atomic_numbers, coordinates, n_sites = get_coordinates(structure)
    coulomb_matrix = np.zeros((n_sites, n_sites))

    for i in range(n_sites):
        for j in range(i):
            if i == j:
                coulomb_matrix[i][j] = 0.5 * np.power(atomic_numbers[i], 2.4)
            else:
                value = (atomic_numbers[i] * atomic_numbers[j]) / np.linalg.norm(coordinates[i] - coordinates[j])
                coulomb_matrix[i][j] = value
                coulomb_matrix[j][i] = value

    return coulomb_matrix

# the idea of representing periodic structure like crystal structure using sine matrix is taken from https://arxiv.org/abs/1503.07406
# code is adapted from https://www.kaggle.com/code/asatoonishi/using-sine-matrix
## it's a modification of Coulomb Matrix representation that has been used to represent organic molecules
## the distance vector here is modified to an alternative coordinate r' where r' = sin^2 r = np.transpose(sin^2 x, sin^2 y, sin^2 z)
def get_sine_matrix(structure:Structure):
    atomic_numbers, coordinates, n_sites = get_coordinates(structure)
    sine_matrix = np.zeros((n_sites, n_sites))
    _lattice_vector = np.transpose([[x for x in structure.lattice.abc]])
    _inverse_lattice_vector = np.linalg.pinv(_lattice_vector)

    for i in range(n_sites):
        for j in range(i):
            if i == j:
                sine_matrix[i][j] = 0.5 * np.power(atomic_numbers[i], 2.4)
            else:
                r_ij = np.dot(_inverse_lattice_vector, coordinates[i] - coordinates[j])
                sin_squared_r_ij = (np.sin(np.pi * r_ij))**2
                value = (atomic_numbers[i] * atomic_numbers[j]) / np.linalg.norm(np.dot(_lattice_vector, sin_squared_r_ij))
                sine_matrix[i][j] = value
                sine_matrix[j][i] = value
    return sine_matrix

def get_eigen_matrix(structure:Structure, matrix_fn:Callable):
    unsorted_CM = matrix_fn(structure)
    eigen = np.linalg.eigvalsh(unsorted_CM)
    eigen = np.sort(eigen)[::-1]    # [::-1] reverse the sorted array and make it into descending order
    return eigen