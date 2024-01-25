import jax.numpy as jnp
from jraph import GraphsTuple
from pymatgen.core import Structure
import numpy as np
from typing import Dict, List

def get_coordinates(structure:Structure):
    atomic_numbers = np.array([x.specie.number for x in structure.sites])
    coordinates = np.array([x.coords for x in structure.sites])
    n_sites = structure.num_sites
    return atomic_numbers, coordinates, n_sites

def get_coulomb_matrix(structure):
        atomic_numbers, coordinates, n_sites = get_coordinates(structure)
        coulomb_matrix = np.zeros((n_sites, n_sites))
        col_index = 0

        for i in range(n_sites):
            for j in range(col_index, n_sites):
                if i == j:
                    coulomb_matrix[i][j] = 0.5 * np.power(atomic_numbers[i], 2.4)
                else:
                    value = (atomic_numbers[i] * atomic_numbers[j]) / np.linalg.norm(coordinates[i] - coordinates[j])
                    coulomb_matrix[i][j] = value
                    coulomb_matrix[j][i] = value
            col_index += 1

        return coulomb_matrix

def get_eigen_matrix(structure):
    unsorted_CM = get_coulomb_matrix(structure)
    eigen = np.linalg.eigvals(unsorted_CM)
    eigen = np.sort(eigen)[::-1]    # [::-1] reverse the sorted array and make it into descending order
    return eigen


def get_node_features(structure:Structure, node_feature_names:List[str]) -> jnp.array:
    n_node_features = len(node_feature_names)
    graph_node_features = np.zeros((structure.num_sites, n_node_features))
    for idx, ele in enumerate(structure.species):
        for i in range(n_node_features):
            graph_node_features[idx][i] = ele.data[node_feature_names[i]]
    return jnp.array(graph_node_features)

def get_edges(structure:Structure, radius:float):
    neighbors = structure.get_neighbor_list(r=radius)
    edge_origin = jnp.array(neighbors[0])
    edge_target = jnp.array(neighbors[1])
    edge_dist = jnp.expand_dims(neighbors[3], 1)
    return edge_origin, edge_target, edge_dist

def structure_to_graph(structure:Structure, node_feature_names:List[str], radius:float=5.0) -> GraphsTuple:
    graph_node_features = get_node_features(structure, node_feature_names)
    edge_origin, edge_target, edge_features = get_edges(structure, radius)
    n_node = structure.num_sites
    n_edge = edge_origin.shape

    lattice_x, lattice_y, lattice_z = structure.lattice.abc
    angle_a, angle_b, angle_c = structure.lattice.angles
    global_context = jnp.array([[lattice_x, lattice_y, lattice_z, angle_a, angle_b, angle_c]])
    return GraphsTuple(
        nodes=graph_node_features,
        edges=edge_features,
        senders=edge_origin,
        receivers=edge_target,
        n_node=n_node,
        n_edge=n_edge,
        globals=global_context
    )



    
