import os
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader  # Updated import to resolve deprecation warning
from torch_geometric.utils import remove_self_loops, coalesce, from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
import pickle

def prune_adjacency_matrix(matrix, top_percent=0.1):
    """
    Prune the adjacency matrix to keep only the top percentage of edges based on weights.
    
    Args:
        matrix (numpy.ndarray): The adjacency matrix.
        top_percent (float): Fraction of top edges to retain (e.g., 0.1 for top 10%).
    
    Returns:
        pruned_matrix (numpy.ndarray): The pruned adjacency matrix.
    """
    # Extract non-zero edge weights
    edge_weights = matrix[matrix > 0].flatten()
    
    if len(edge_weights) == 0:
        return matrix  # No edges to prune
    
    # Determine the number of edges to keep
    num_edges_to_keep = int(len(edge_weights) * top_percent)
    num_edges_to_keep = max(num_edges_to_keep, 1)  # Ensure at least one edge is kept
    
    # Find the threshold weight
    sorted_weights = np.sort(edge_weights)[::-1]  # Sort descendingly
    threshold = sorted_weights[num_edges_to_keep - 1]
    
    # Prune the adjacency matrix
    pruned_matrix = np.where(matrix >= threshold, matrix, 0)
    
    return pruned_matrix

def compute_nodal_measures(G):
    """
    Compute nodal graph measures to be used as node features.
    
    Args:
        G (networkx.Graph): The graph object.
    
    Returns:
        features (numpy.ndarray): Array of shape [num_nodes, num_features].
    """
    num_nodes = G.number_of_nodes()
    
    # Degree
    degrees = np.array([val for (node, val) in G.degree()])
    
    # Closeness Centrality
    closeness = np.array([val for (node, val) in nx.closeness_centrality(G).items()])
    
    # Betweenness Centrality
    betweenness = np.array([val for (node, val) in nx.betweenness_centrality(G).items()])
    
    # Clustering Coefficient
    clustering = np.array([val for (node, val) in nx.clustering(G).items()])
    
    # Combine all measures into a feature matrix
    # Shape: [num_nodes, 4]
    features = np.vstack((degrees, closeness, betweenness, clustering)).T
    
    return features

def load_data_with_graph_attributes(base_dir, top_percent=0.1):
    """
    Load matrices from .xlsx files organized in group-specific folders and construct graph data objects
    with nodal graph measures as node features.
    
    Args:
        base_dir (str): Path to the base directory containing group folders.
    
    Returns:
        data_list (list of Data): List containing Data objects with graph attributes and labels.
    """
    data_list = []
    
    # Initialize LabelEncoder to convert folder names to numerical labels
    le = LabelEncoder()
    
    # List all subdirectories in the base directory
    groups = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if not groups:
        print(f"No group folders found in {base_dir}.")
        return data_list
    
    # Fit LabelEncoder on group names
    le.fit(groups)
    
    print(f"Found groups: {groups}")
    print(f"Assigned labels: {list(le.classes_)}")
    
    for group in groups:
        group_path = os.path.join(base_dir, group)
        label = le.transform([group])[0]  # Numerical label
        
        # List all .xlsx files in the group folder
        xlsx_files = [f for f in os.listdir(group_path) if f.endswith('.xlsx')]
        
        if not xlsx_files:
            print(f"No .xlsx files found in {group_path}. Skipping this group.")
            continue
        
        for file in xlsx_files:
            file_path = os.path.join(group_path, file)
            try:
                # Read the Excel file into a pandas DataFrame
                df = pd.read_excel(file_path, header=None)  # Assuming no headers
                
                # Convert DataFrame to NumPy array
                matrix = df.values.astype(float)
                
                # Validate the matrix
                if matrix.shape[0] != matrix.shape[1]:
                    print(f"Matrix in {file_path} is not square. Skipping this file.")
                    continue
                
                num_nodes = matrix.shape[0]
                matrix = prune_adjacency_matrix(matrix, top_percent=top_percent)
                # Create a NetworkX graph from the adjacency matrix
                G = nx.from_numpy_array(matrix)
                
                # Compute nodal graph measures
                node_features = compute_nodal_measures(G)  # Shape: [num_nodes, 4]
                
                # Convert node features to torch tensor
                x = torch.tensor(node_features, dtype=torch.float)  # Shape: [num_nodes, 4]
                # Create adjacency matrix as a sparse COO matrix
                adjacency_matrix = coo_matrix(matrix)
                
                # Extract edge_index and edge_attr using PyTorch Geometric utility
                edge_index, edge_attr = from_scipy_sparse_matrix(adjacency_matrix)
                
                # Remove self-loops (redundant if already removed above)
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                
                # Coalesce to remove duplicate edges and sort
                edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)
                
                # Positional encoding (identity matrix)
                pos = torch.eye(num_nodes, dtype=torch.float)
                
                # Create label tensor
                y = torch.tensor([label], dtype=torch.long)
                
                # Construct the Data object
                data = Data(
                    x=x,  # Node features with gradient tracking
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    pos=pos
                )                                
                data_list.append(data)
                
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
    
    # Save the LabelEncoder for future use
    try:
        with open(os.path.join(base_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(le, f)
        print(f"Label encoder saved at {os.path.join(base_dir, 'label_encoder.pkl')}")
    except Exception as e:
        print(f"Failed to save LabelEncoder: {e}")
    
    return data_list

def load_label_encoder(base_dir):
    """
    Load the saved LabelEncoder from the base directory.
    
    Args:
        base_dir (str): Path to the base directory containing the label_encoder.pkl.
    
    Returns:
        label_encoder (LabelEncoder): Loaded LabelEncoder instance.
    """
    try:
        with open(os.path.join(base_dir, 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
        print("Label encoder loaded successfully.")
        return label_encoder
    except Exception as e:
        print(f"Failed to load LabelEncoder: {e}")
        return None

