import numpy as np
import torch
from scipy.spatial import Delaunay, KDTree
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

BSE_FEATURES = 1
EDS_FEATURES = 64

BSE_HEIGHT = 0
EDS_HEIGHT = 5  # the exact does not really matter, it just should not be the same as BSE_HEIGHT


def create_edsbse_graph(bse: np.ndarray, eds: np.ndarray, phases: np.ndarray, eds_mask: np.ndarray) -> Data:
    """Creates a graph representation for the fusion of BSE and EDS data.

    Args:
        bse (ndarray): The BSE data.
        eds (ndarray): The EDS data.
        phases (ndarray): The phase information (ground truth).
        eds_frac (float, optional): The fraction of EDS nodes to select randomly. Defaults to 0.2.
        bse_frac (float, optional): The fraction of BSE nodes to select randomly. Defaults to 1.0.
        with_node_type (bool, optional): Whether to include the node type as an attribute. Defaults to False.
        eds_mask (ndarray, optional): The mask for selecting EDS nodes. If specified, eds_frac is ignored. Defaults to None.

    Returns:
        graph (Data): The graph representation of the fusion of BSE and EDS data.
    """
    if eds_mask.sum() < 3:
        raise ValueError("At least 3 EDS points are required to create a graph.")

    # prepare node features
    flatbse = bse.reshape(-1, 1)
    y = phases.reshape(-1, 1)
    height, width = bse.shape

    # merge shapes and fill with zeroes where information missing
    eds_nodes_features = np.pad(eds, ((0, 0), (BSE_FEATURES, 0)), mode="constant", constant_values=0)
    bse_nodes_features = np.pad(flatbse, ((0, 0), (0, EDS_FEATURES)), mode="constant", constant_values=0)

    selected_y = torch.tensor(np.concatenate([y, y[eds_mask.ravel()]]))
    merged_features = np.concatenate([bse_nodes_features, eds_nodes_features], axis=0)

    merged_features = torch.tensor(merged_features, dtype=torch.float)

    # infer coordinates
    bse_y_coords, bse_x_coords = np.mgrid[0:height, 0:width]
    eds_y_coords, eds_x_coords = np.where(eds_mask)

    bse_coords = np.stack([bse_x_coords.ravel(), bse_y_coords.ravel(), np.full(bse_x_coords.size, BSE_HEIGHT)], axis=1)
    eds_coords = np.stack([eds_x_coords.ravel(), eds_y_coords.ravel(), np.full(eds_x_coords.size, EDS_HEIGHT)], axis=1)
    merged_coords = np.concatenate([bse_coords, eds_coords], axis=0)

    # crete edges
    bse_edges, bse_attrs = triangulate_bse(height, width)
    eds_edges, eds_edge_attrs = triangulate_eds(eds_coords, height, width)
    joint_edges, joint_attrs = tesselate_joint(bse_coords, eds_coords, height, width)
    edges = np.concatenate([bse_edges, eds_edges, joint_edges], axis=0)
    edge_attrs = np.concatenate([bse_attrs, eds_edge_attrs, joint_attrs], axis=0)

    edge_attrs = torch.tensor(edge_attrs, dtype=torch.float32)
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    # fill missing edges to create undirected graph
    edge_index, edge_attrs = to_undirected(edge_index, edge_attrs)

    graph = Data(
        x=merged_features,
        y=selected_y,
        pos=torch.tensor(merged_coords, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attrs,
    )
    graph.validate(raise_on_error=True)
    return graph


def tesselate_joint(bse_coords, eds_coords, height, width):
    """Create inter-modality edges.

    Args:
        bse_coords (array-like): The base coordinates.
        eds_coords (array-like): The edge coordinates.
        height (int): The height of the joint.
        width (int): The width of the joint.

    Returns:
        tuple: A tuple containing the joint edges and joint attributes.
            - joint_edges (list): A list of tuples representing the joint edges.
            - joint_attrs (ndarray): An ndarray object representing the joint attributes.
    """
    bse_tree = KDTree(bse_coords)

    joint_edges = []
    joint_attrs = []

    dists, indices = bse_tree.query(eds_coords)
    joint_attrs = dists
    indices = np.atleast_1d(indices)
    joint_edges = list(zip(indices, range(width * height, width * height + len(eds_coords)), strict=True))

    eds_tree = KDTree(eds_coords)
    dists, indices = eds_tree.query(bse_coords)
    joint_attrs = np.concatenate([joint_attrs, dists])
    joint_edges += list(zip(range(width * height), indices + width * height, strict=True))
    return joint_edges, joint_attrs


def triangulate_eds(eds_coords, height, width):
    """Triangulates the given EDS coordinates using Delaunay triangulation or KNN graph as a fallback.

    Args:
        eds_coords (ndarray): The EDS coordinates.
        height (int): The height of the image.
        width (int): The width of the image.

    Returns:
        tuple: A tuple containing the edges and edge attributes of the triangulated EDS.
    """
    eds_edges = []
    eds_edge_attrs = []
    try:
        tri = Delaunay(eds_coords[:, :2], qhull_options="Qbb Qc Qz Q12")
        indptr, indices = tri.vertex_neighbor_vertices
        for i in range(len(indptr) - 1):
            for j in range(indptr[i], indptr[i + 1]):
                eds_edges.append((i, indices[j]))
                eds_edge_attrs.append(np.linalg.norm(eds_coords[i] - eds_coords[indices[j]]))
        eds_edges = np.array(eds_edges) + width * height

    except Exception as e:  # Initial simplex is flat or something
        print("Delaunay triangulation failed, using KNN graph as a fallback")
        csr = kneighbors_graph(eds_coords, 2, mode="distance")
        eds_edges = np.array(csr.nonzero()).T + width * height
        eds_edge_attrs = csr.data

    return eds_edges, eds_edge_attrs


def triangulate_bse(height, width):
    """Create a graph representing full BSE image.

    Args:
        height (int): The height of the image.
        width (int): The width of the image.

    Returns:
        tuple: A tuple containing two lists - `bse_edges` and `bse_attrs`.
            - `bse_edges` (list): A list of tuples representing the edges between neighboring pixels.
            - `bse_attrs` (list): A list of edge attributes corresponding to each edge.

    """
    bse_edges = []
    bse_edge_attrs = []

    STRAIGHT = 1.0
    DIAGONAL = np.sqrt(2)

    # Add edges for neighboring pixels (8-connected neighbors)
    for y in range(height):
        for x in range(width):
            # Check for valid coordinates within image boundaries
            if x > 0:
                bse_edges.append((y * width + x, y * width + x - 1))
                bse_edge_attrs.append(STRAIGHT)
            if x < width - 1:
                bse_edges.append((y * width + x, y * width + x + 1))
                bse_edge_attrs.append(STRAIGHT)
            if y > 0:
                bse_edges.append((y * width + x, (y - 1) * width + x))
                bse_edge_attrs.append(STRAIGHT)
            if y < height - 1:
                bse_edges.append((y * width + x, (y + 1) * width + x))
                bse_edge_attrs.append(STRAIGHT)

            if x > 0 and y > 0:
                bse_edges.append((y * width + x, (y - 1) * width + x - 1))
                bse_edge_attrs.append(DIAGONAL)
            if x < width - 1 and y > 0:
                bse_edges.append((y * width + x, (y - 1) * width + x + 1))
                bse_edge_attrs.append(DIAGONAL)
            if x > 0 and y < height - 1:
                bse_edges.append((y * width + x, (y + 1) * width + x - 1))
                bse_edge_attrs.append(DIAGONAL)
            if x < width - 1 and y < height - 1:
                bse_edges.append((y * width + x, (y + 1) * width + x + 1))
                bse_edge_attrs.append(DIAGONAL)
    return bse_edges, bse_edge_attrs


class ToGraphEDS(object):
    """Converts BSE and EDS data into a graph representation. This class is mainly used for training.

    Args:
        with_node_type (bool): Flag indicating whether to include node types in the graph.
        eds_sparsity_range (tuple): Range of sparsity values for EDS data.

    Returns:
        Data: Graph representation of the input BSE and EDS data.

    """

    def __init__(self, eds_sparsity_range) -> None:
        self.eds_sparsity_range = eds_sparsity_range

    def __call__(self, bse, eds, phases) -> Data:
        """Generates a fused graph by randomly masking the EDS data according to a sparsity range.

        Args:
            bse (np.ndarray): Base data array.
            eds (np.ndarray): Dense EDS data array. It is assumed that it is already in its reduced form.
            phases (Any): Phase information associated with the data.

        Returns:
            Data: A fused graph object created from the masked EDS and base data.
        """
        ea, eb = self.eds_sparsity_range
        eds_frac = (eb - ea) * np.random.random_sample() + ea

        mask = np.random.random_sample(size=bse.shape) < eds_frac

        return create_edsbse_graph(bse, eds[mask], phases, mask)
