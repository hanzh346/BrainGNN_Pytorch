import pandas as pd
import numpy as np
import scipy.io
import os
import timeit
import networkx as nx
# Assume dataTable.csv is your dataset containing subject IDs, diagnosis info, etc.
dataTable = pd.read_csv('/home/hang/GitHub/BrainGNN_Pytorch/data/filtered_selectedDataUnique_merged_ADNI.csv')
mat_files_dir = "/media/hang/EXTERNAL_US/Data/1_HANG_FDG_PET/ADNI_Second_organized/KDE_Results/" 
graph_measure_path = '/media/hang/EXTERNAL_US/Data/1_HANG_FDG_PET/ADNI_Second_organized/organized/KDE_Results/reorgnized_AllMeasuresAndDiagnosisByThreshold_DISTANCE.mat'
import os
import numpy as np
import pandas as pd
import torch
import networkx as nx
from scipy.io import loadmat
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, remove_self_loops
from torch_sparse import coalesce
from os import listdir
import multiprocessing
from functools import partial
def extract_features(file_path, field_name='scaledMahalDistMatrix'):
    """
    Placeholder function for extracting features from a .mat file.
    Implement according to your specific requirements.
    """
    mat = loadmat(file_path)
    return mat[field_name]

def apply_threshold(features, percentile=20):
    """
    Placeholder function for applying thresholding to features.
    Implement based on your specific needs.
    """
    threshold = np.percentile(features, percentile)
    features[features < threshold] = 0
    return features

def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row.numpy()])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Adjust edge indices to start from zero for each graph
    #data.edge_index[0, :] -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice, 'x': node_slice, 'edge_attr': edge_slice, 'y': torch.arange(0, len(batch) + 1, dtype=torch.long), 'pos': node_slice}
    return data, slices
import torch
import numpy as np

def process_subject(subject_id, binary_label, mat_files_dir):
    """
    Process a single subject's graph data for classification.
    """
    mat_file_path = os.path.join(mat_files_dir, f"{subject_id}_ScaledMahalanobisDistanceMatrix.mat")
    par_corr_file_path = os.path.join(mat_files_dir, f"{subject_id}_partial_correlation_KDE.mat")

    if not os.path.exists(mat_file_path) or not os.path.exists(par_corr_file_path):
        return None

    node_features = extract_features(mat_file_path)  # This should be a numpy array
    node_features = apply_threshold(node_features)  # Still a numpy array

    edge_features = extract_features(par_corr_file_path, field_name='partial_corr')
    edge_features = apply_threshold(edge_features)  # Still a numpy array

    num_nodes = node_features.shape[1]

    G = nx.from_numpy_array(edge_features)
    A = nx.to_scipy_sparse_array(G)  # Use scipy sparse matrix for compatibility
    adj = A.tocoo()

    edge_att = np.array([edge_features[adj.row[i], adj.col[i]] for i in range(len(adj.row))])

    edge_index = np.stack([adj.row, adj.col], axis=0)

    # Convert all numpy arrays to tensors before returning
    return {
        'edge_index': torch.tensor(edge_index, dtype=torch.long),
        'edge_att': torch.tensor(edge_att, dtype=torch.float),
        'att': torch.tensor(node_features, dtype=torch.float),
        'y': torch.tensor([binary_label], dtype=torch.long),
        'num_nodes': num_nodes
    }

def read_data(dataTable, class_pairs, mat_files_dir):
    all_results = []
    batch = []
    pseudo = []
    edge_index_list = []
    edge_att_list = []
    y_list = []
    att_list = []
    start = timeit.default_timer()

    for class_0_labels, class_1_labels in class_pairs:
        if not isinstance(class_1_labels, list):
            class_1_labels = [class_1_labels]

        dataTable['binary_labels'] = dataTable.apply(lambda row:
            0 if row['DX_bl'] in class_0_labels and row['AV45'] < 1.11 else
            1 if row['DX_bl'] in class_1_labels and row['AV45'] >= 1.11 else np.nan, axis=1)
        filtered_data = dataTable.dropna(subset=['binary_labels'])
        filtered_data = filtered_data.drop_duplicates(subset='PTID',keep='first')
        subjects = [(row['PTID'], int(row['binary_labels'])) for _, row in filtered_data.iterrows()]
        args = [(sub_id, label, mat_files_dir) for sub_id, label in subjects]

        cores = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=int(cores / 2)) as pool:
            func = partial(process_subject)
            results = pool.starmap(func, args)

        pool.close()
        pool.join()

        for result in results:
            if result is None:
                continue
            edge_att_list.append(result['edge_att'].numpy())  # Convert tensor to numpy for concatenation
            edge_index_list.append(result['edge_index'].numpy())
            att_list.append(result['att'].numpy())
            y_list.append(result['y'].numpy())
            num_nodes = result['num_nodes']
            batch.append(np.full((num_nodes,), fill_value=len(all_results), dtype=int))  # Batch index for all nodes in this graph
            all_results.append(result)
            pseudo.append(np.eye(num_nodes))  # Assuming pseudo-positional info if needed

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    # Convert lists to tensors for PyTorch Geometric
    edge_index_tensor = torch.from_numpy(np.concatenate(edge_index_list, axis=1))
    edge_att_tensor = torch.from_numpy(np.concatenate(edge_att_list))
    att_tensor = torch.from_numpy(np.concatenate(att_list))
    y_tensor = torch.from_numpy(np.concatenate(y_list))
    batch_tensor = torch.from_numpy(np.concatenate(batch))
    pseudo_tensor = torch.from_numpy(np.concatenate(pseudo, axis=0))

    data = Data(x=att_tensor, edge_index=edge_index_tensor, edge_attr=edge_att_tensor, y=y_tensor, pos=pseudo_tensor)
    data, slices = split(data, batch_tensor)

    return data, slices

# class_pairs = [
#     (['CN'], ['MCI', 'AD']),  # Example class pair
#     # Add more class pairs as needed
# ]


# # Now, `all_data` contains a list of Data objects ready for GNN processing.

# class_pairs = [
#      (['CN', 'SMC'], ['EMCI', 'LMCI']),
#      (['CN', 'SMC'], 'AD'),
#      (['CN', 'SMC'], ['CN', 'SMC']),  # Assuming 'CN ab+' is represented like this in the 'DX_bl' column
#     # (['EMCI', 'LMCI'],'AD'),
# ]
