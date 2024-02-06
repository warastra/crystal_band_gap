import jax.numpy as jnp
from jraph import GraphsTuple
from pymatgen.core import Structure
import numpy as np
from typing import Dict, List, Optional, Union, Callable

class GraphDataPoint:
    """
    input_graph: GraphsTuple instance of the crystal/molecule
    target: value of target property to predict as jnp.ndarray with size (1)
    id: string identifier, for data from materials project this will be the material_id
    """
    def __init__(self, input_graph:GraphsTuple, target:jnp.ndarray, id:Optional[str]):
        self.input_graph = input_graph
        self.target = target
        self.id = id

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





def get_node_features(structure:Structure, node_feature_names:List[str]) -> jnp.array:
    n_node_features = len(node_feature_names)
    graph_node_features = np.empty((structure.num_sites, n_node_features), dtype=np.int64)
    for idx, ele in enumerate(structure.species):
        for i in range(n_node_features):
            graph_node_features[idx][i] = np.int64(ele.data[node_feature_names[i]])
    return jnp.array(graph_node_features)

def get_edges(structure:Structure, radius:float):
    neighbors = structure.get_neighbor_list(r=radius)
    edge_origin = jnp.array(neighbors[0])
    edge_target = jnp.array(neighbors[1])
    edge_dist = jnp.expand_dims(neighbors[3].astype(np.float64), 1)
    return edge_origin, edge_target, edge_dist

def structure_to_graph(structure:Structure, node_feature_names:List[str], radius:float=5.0) -> GraphsTuple:
    graph_node_features = get_node_features(structure, node_feature_names)
    edge_origin, edge_target, edge_features = get_edges(structure, radius)
    n_node = jnp.array([structure.num_sites])
    n_edge = jnp.array([edge_origin.shape[0]])

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

def get_graph_dataset(
      structures:List[Structure], 
      structure_to_graph_fn:Callable, 
      targets:Union[np.ndarray, List[Union[int, float]]], 
      ids:Optional[List[str]]
    ):
    dataset = []
    if ids is None:
        ids = [None] * len(targets)
    for i in range(len(targets)):
      structure = Structure.from_dict(structures[i])
      input_graph = structure_to_graph_fn(structure)

      target = jnp.array([targets[i]])
      dataset.append(GraphDataPoint(input_graph=input_graph, target=target, id=ids[i]))
    
    return dataset