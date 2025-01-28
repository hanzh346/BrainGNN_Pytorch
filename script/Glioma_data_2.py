import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample  # For resampling
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
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

def load_node_features(file_path):
    """
    Load node features from an Excel file.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        node_features (numpy.ndarray): Array of shape [num_nodes, num_features].
    """
    try:
        # Read the Excel file into a pandas DataFrame
        # Assuming the first row contains headers: ['ROI', 'Strength', 'GlobalEfficiency', ...]
        df = pd.read_excel(file_path, header=0)
        
        # Validate required columns
        # required_measures = ['Strength', 'GlobalEfficiency', 'CorePeriphery',
                            #  'EigenVectorCentrality', 'Degree', 'PathLength']
        required_measures = ['Strength']
        if not all(measure in df.columns for measure in required_measures):
            raise ValueError(f"Missing required measures in {file_path}.")
        
        # If 'ROI' column exists, you can choose to drop it or keep it
        if 'ROI' in df.columns:
            df = df.drop(columns=['ROI'])
        
        # Convert DataFrame to NumPy array
        node_features = df['Strength'].values.astype(float)  # Shape: [num_nodes, num_measures]
        if len(node_features.shape) == 1:
            node_features = np.expand_dims(node_features,1) 

        return node_features

    except Exception as e:
        print(f"Failed to load node features from {file_path}: {e}")
        return None

def load_data_with_graph_attributes(base_dir, density, top_percent=0.15, balance=True, random_state=42):
    """
    Load node features and adjacency matrices from Excel files for a specified density and construct graph data objects.
    Optionally balance the dataset by downsampling majority classes.

    Args:
        base_dir (str): Path to the base directory containing group folders.
        density (float): Density level to process (e.g., 0.15 for 15%).
        top_percent (float): Fraction of top edges to retain in the adjacency matrix.
        balance (bool): Whether to balance the dataset by downsampling majority classes.
        random_state (int): Random seed for reproducibility.

    Returns:
        data_list (list of Data): List containing Data objects with graph attributes and labels.
    """
    data_list = []
    labels_list = []  # To store labels for balancing

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

        # Define the density string for matching filenames
        density_str = f"Density_{density:.2f}"

        # List all .xlsx files in the group folder matching the current density
        # Assuming filenames are in the format <SubjectID>_GraphMeasures_Density_<Density>.xlsx
        # and adjacency matrices are named as <SubjectID>.xlsx
        graph_measure_pattern = f"_GraphMeasures_Density_{density:.2f}.xlsx"
        
        # Extract SubjectIDs by removing the pattern from filenames
        xlsx_files = [f for f in os.listdir(group_path) 
                      if f.endswith(graph_measure_pattern)]
        
        if not xlsx_files:
            print(f"No .xlsx files found for density {density} in {group_path}. Skipping this group.")
            continue

        for file in xlsx_files:
            file_path = os.path.join(group_path, file)
            try:
                # Extract SubjectID from filename
                # Assuming filename format: <SubjectID>_GraphMeasures_Density_<Density>.xlsx
                subject_id = file.replace(graph_measure_pattern, "")
                
                # Define the adjacency matrix file path
                adj_file = f"{subject_id}.xlsx"
                adj_file_path = os.path.join(group_path, adj_file)
                
                if not os.path.isfile(adj_file_path):
                    print(f"Adjacency matrix file {adj_file_path} not found. Skipping Subject ID: {subject_id}.")
                    continue
                
                # Load adjacency matrix
                df = pd.read_excel(adj_file_path, header=None)
                
                # Convert DataFrame to NumPy array
                adjacency_matrix = df.values.astype(float)  # Shape: [num_nodes, num_nodes]
                if adjacency_matrix is None:
                    continue  # Skip if failed to load

                # Prune the adjacency matrix based on the density
                pruned_matrix = prune_adjacency_matrix(adjacency_matrix, top_percent=top_percent)

                # Load node features
                node_features = load_node_features(file_path)
                if node_features is None:
                    continue  # Skip if failed to load

                num_nodes_adj = adjacency_matrix.shape[0]
                num_nodes_features = node_features.shape[0]

                # Validate that the number of nodes matches
                if num_nodes_adj != num_nodes_features:
                    print(f"Node count mismatch for Subject ID: {subject_id} at density {density}.")
                    print(f"Adjacency matrix nodes: {num_nodes_adj}, Node features: {num_nodes_features}. Skipping.")
                    continue

                num_nodes = num_nodes_adj  # Both are equal

                # Convert node features to torch tensor
                x = torch.tensor(node_features, dtype=torch.float)  # Shape: [num_nodes, num_features]
                if torch.isnan(x).any() or torch.isinf(x).any():
                    logging.warning(f"NaN or Inf detected in node features for Subject ID: {subject_id}. Replacing with zeros.")
                    print(f"NaN or Inf detected in node features for Subject ID: {subject_id}. Replacing with zeros.")
                    continue
                # Create adjacency matrix as a sparse COO matrix
                adjacency_sparse = coo_matrix(pruned_matrix)

                # Extract edge_index and edge_attr using PyTorch Geometric utility
                edge_index, edge_attr = from_scipy_sparse_matrix(adjacency_sparse)

                # Remove self-loops (redundant if already removed above)
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

                # Coalesce to remove duplicate edges and sort
                edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)
                if torch.isnan(edge_attr).any() or torch.isinf(edge_attr).any():
                    logging.warning(f"NaN or Inf detected in edge attributes for Subject ID: {subject_id}. Replacing with zeros.")
                    print(f"NaN or Inf detected in edge attributes for Subject ID: {subject_id}. Replacing with zeros.")
                    continue
                # Positional encoding (identity matrix)
                pos = torch.eye(num_nodes, dtype=torch.float)

                # Create label tensor
                y = torch.tensor([label], dtype=torch.long)
                print(x.shape)
                # Construct the Data object
                data = Data(
                    x=x,  # Node features
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    pos=pos,
                    id = subject_id,
                    original_edge_attr = torch.flatten(torch.tensor(adjacency_matrix))
                )

                data_list.append(data)
                labels_list.append(label)  # Store label for balancing



            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

    # Optional: Balance the dataset by downsampling majority classes
    if balance and data_list:
        print("\nBalancing the dataset by downsampling majority classes...")
        # Convert labels_list to a NumPy array for easier indexing
        labels_array = np.array(labels_list)

        # Find the minimum class count
        unique_classes, class_counts = np.unique(labels_array, return_counts=True)
        min_count = class_counts.min()
        print(f"Class distribution before balancing: {dict(zip(unique_classes, class_counts))}")
        print(f"Minimum class count: {min_count}")

        balanced_indices = []
        for cls in unique_classes:
            cls_indices = np.where(labels_array == cls)[0]
            # If the class has fewer samples than min_count, keep all samples
            if len(cls_indices) <= min_count:
                selected_indices = cls_indices
            else:
                # Randomly select min_count samples without replacement
                selected_indices = np.random.choice(cls_indices, size=min_count, replace=False)
            balanced_indices.extend(selected_indices)
            print(f"Class {cls}: selected {len(selected_indices)} samples.")

        # Shuffle the balanced indices to mix classes
        np.random.seed(random_state)
        np.random.shuffle(balanced_indices)

        # Create the balanced data_list
        balanced_data_list = [data_list[i] for i in balanced_indices]
        print(f"Dataset size after balancing: {len(balanced_data_list)}")
    else:
        balanced_data_list = data_list
        if data_list:
            print(f"Dataset size without balancing: {len(balanced_data_list)}")
        else:
            print("No data to balance.")

    # Save the LabelEncoder for future use
    try:
        with open(os.path.join(base_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(le, f)
        print(f"Label encoder saved at {os.path.join(base_dir, 'label_encoder.pkl')}")
    except Exception as e:
        print(f"Failed to save LabelEncoder: {e}")

    return balanced_data_list

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
#Define the base directory containing group-specific folders
base_directory = '/media/hang/EXTERNAL_US/Data/glioma/data/organized data/pre_surgery'

# Define the density you want to process
desired_density = 0.15  # For example, 15%

# Load data for the specified density
data_list = load_data_with_graph_attributes(
    base_dir=base_directory,
    density=desired_density,
    top_percent=0.15,  # Adjust as needed
    balance=True,       # Set to False if you do not want to balance the dataset
    random_state=42     # For reproducibility
)
# Initialize lists to store problematic IDs
nan_ids = []
inf_ids = []
# Iterate through each data point in the data_list
for data in data_list:
    features = data.x
    attr = data.edge_attr
    
    # Check for NaN values
    if torch.isnan(features).any() or torch.isnan(attr).any():
        nan_ids.append(data.id)
    
    # Check for Inf values
    elif torch.isinf(features).any() or torch.isinf(attr).any():
        inf_ids.append(data.id)

# Report the findings
if nan_ids:
    print(f"Data points with NaN values: {nan_ids}")
else:
    print("No NaN values found in data.")

if inf_ids:
    print(f"Data points with Inf values: {inf_ids}")
else:
    print("No Inf values found in data.")
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
    # Optional: Standardize features
    # scaler = StandardScaler()
    # features = scaler.fit_transform(features)
    return features