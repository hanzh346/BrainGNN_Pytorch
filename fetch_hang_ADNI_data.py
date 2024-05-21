import pandas as pd
import numpy as np
import os
import timeit
import networkx as nx
import os
import numpy as np
import pandas as pd
import torch
import networkx as nx
from scipy.io import loadmat
from torch_geometric.data import Data,  Batch
from torch_geometric.utils import  remove_self_loops
import multiprocessing
import glob
from torch_sparse import coalesce
from functools import partial
def extract_features(file_path, field_name='scaledMahalDistMatrix'):
    """
    Placeholder function for extracting features from a .mat file.
    Implement according to your specific requirements.
    """
    mat = loadmat(file_path)
    return mat[field_name]

def apply_threshold(matrix, percentile):
    # Flatten the matrix and sort it
    sorted_matrix = np.sort(matrix)
    # Calculate the index for the desired percentile
    threshold_index = round(len(sorted_matrix) * (100 - percentile) / 100)  # Adjust for zero indexing

    # Get the threshold value using the sorted matrix
    threshold_value = sorted_matrix[threshold_index]
    # Apply thresholding
    thresholded_matrix = np.where(matrix > threshold_value, matrix, 0)
    return thresholded_matrix


def process_subject(subject_id, binary_label, mat_files_dir, percentile, connectome):
    """
    Process a single subject's graph data for classification, including binary label.
    """
    mat_file_path = os.path.join(mat_files_dir, f"{subject_id}_{connectome}.mat")
    par_corr_file_path = os.path.join(mat_files_dir, f"{subject_id}_partial_correlation_KDE.mat")

    if not os.path.exists(mat_file_path) or not os.path.exists(par_corr_file_path):
        print(f"Files for subject {subject_id} are missing.")
        return None
    field_name = [
        'scaledMahalDistMatrix' if connectome == 'ScaledMahalanobisDistanceMatrix' else 
        'zScoreConnectivityMatrix' if connectome == 'Z_scoring' else
        'correlationMatrix' if connectome == 'K_correlation' else
        'JSdivMatrix' ][0]
    # field_name = [
    #     'scaledMahalDistMatrix' if connectome == 'ScaledMahalanobisDistanceMatrix' else 
    #     'zScoreConnectivityMatrix' if connectome == 'Z_scoring' else
    #     'correlationsubject' if connectome == 'correted_correlation' else
    #     'JSsubject' ][0]

    node_features = apply_threshold(extract_features(mat_file_path, field_name=field_name), percentile=percentile)
    edge_features = apply_threshold(extract_features(par_corr_file_path, field_name='partial_corr'), percentile=percentile)

    if node_features is None or edge_features is None:
        print(f"Feature extraction failed for {subject_id}.")
        return None

    num_nodes = node_features.shape[0]
    G = nx.from_numpy_array(edge_features)
    A = nx.to_scipy_sparse_array(G, format='coo')
    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = node_features[adj.row[i], adj.col[i]]
    edge_index = np.stack([A.row, A.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                    num_nodes)
    pos = torch.eye(num_nodes)  # Using an identity matrix for positional data

    return Data(x=torch.tensor(node_features, dtype=torch.float),
                edge_index=edge_index,
                edge_attr=edge_att,
                y=torch.tensor([binary_label], dtype=torch.long),
                pos=pos)

def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if data.pos is not None:
        slices['pos'] = node_slice

    return data, slices
# def read_data(dataTable, class_pairs, mat_files_dir,connectome, num_classes,percentile=10):
#     all_data = []

# #for class_0_labels, class_1_labels in class_pairs:
#     class_0_labels = class_pairs[0]
#     class_1_labels = class_pairs[1]
    
#     if num_classes ==2:
#         # dataTable['binary_labels'] = dataTable.apply(lambda row:
#         #     0 if row['DX_bl'] in class_0_labels and row['AV45'] < 1.11 else
#         #     1 if row['DX_bl'] in class_1_labels and row['AV45'] >= 1.11 else np.nan, axis=1)
#         dataTable['binary_labels'] = dataTable.apply(lambda row:
#             0 if row['DX_bl'] in class_0_labels and row['SUMMARYSUVR_WHOLECEREBNORM'] < 1.11 else
#             1 if row['DX_bl'] in class_1_labels and row['SUMMARYSUVR_WHOLECEREBNORM'] >= 1.11 else np.nan, axis=1)
        
#     else:
#         class_2_labels = class_pairs[2]  
#         class_3_labels = class_pairs[3]  
#         dataTable['binary_labels'] = dataTable.apply(lambda row:
#             0 if row['DX_bl'] in class_0_labels and row['AV45'] < 1.11 else
#             1 if row['DX_bl'] in class_1_labels and row['AV45'] >= 1.11 else
#             2 if row['DX_bl'] in class_2_labels and row['AV45'] >= 1.11 else
#             3 if row['DX_bl'] in class_3_labels and row['AV45'] >= 1.11 else np.nan, axis=1)
    

#     filtered_data = dataTable.dropna(subset=['binary_labels'])
#     filtered_data = filtered_data.drop_duplicates(subset='PTID', keep='first')
    
#     for _, row in filtered_data.iterrows():
#         subject_id = row['PTID']
#         binary_label = int(row['binary_labels'])
#         data = process_subject(subject_id, binary_label, mat_files_dir, percentile=percentile,connectome=connectome)
#         if data is not None:
#             all_data.append(data)

#     return all_data#, batch_torch
def read_data( class_pairs, mat_files_dir,connectome, num_classes,baseline=True,percentile=10):
    all_data = []
    ADNI_merge = pd.read_csv('/media/hang/EXTERNAL_US/Data/1_HANG_FDG_PET/ADNIMERGE_30Jan2024.csv')
    UCBERKLEY_PET = pd.read_excel('/home/hang/GU/Project/AD_classification_synthesis/data/ADNI/UCBERKELEYAV45_01_14_21.xlsx')


    # Path to the raw data
    data_list_included = glob.glob('/media/hang/EXTERNAL_US/Data/1_HANG_FDG_PET/ADNI_Second_organized/KDE_Results/raw/*meanSUVR.mat')

    # Extract subject IDs from file names
    subject_ids_included = [filename.split('/')[-1].split('_meanSUVR')[0] for filename in data_list_included]




    for id in subject_ids_included:
        # Get the diagnosis values sorted by exam date for the subject
        diagnosis_values = ADNI_merge[ADNI_merge['PTID'] == id].sort_values('EXAMDATE')['DX'].dropna().values
        
        # Check if there is more than one unique diagnosis
        if len(set(diagnosis_values)) > 1:
            continue

        # Extract RID from the subject ID
        rid = id.split('_')[-1]
        print(f"Extracted RID: {rid}")

        # Check the SUVR values for the extracted RID
        suvr_values = UCBERKLEY_PET[UCBERKLEY_PET['RID'].astype(str) == rid]['SUMMARYSUVR_COMPOSITE_REFNORM']
        suvr_values_bl = UCBERKLEY_PET.loc[(UCBERKLEY_PET['RID'].astype(str) == rid) & (UCBERKLEY_PET['VISCODE2'] == 'bl'), 'SUMMARYSUVR_WHOLECEREBNORM']
        # Skip if SUVR values are empty or have mixed values for threshold
        if np.size(diagnosis_values) == 0 or  suvr_values.empty or suvr_values_bl.empty or len(set(suvr_values > 0.79)) > 1:
            continue

        # Skip if there are fewer than 2 diagnosis values or SUVR values
        # if len(diagnosis_values) < 2 or len(suvr_values) < 2:
        #     continue

        # Determine the diagnosis and amyloid-beta status

        diagnosis = diagnosis_values[0]
        ab_long = (suvr_values > 0.79).astype(int).values[0]
        ab_bl = (suvr_values_bl > 1.11).astype(int).values
        if ab_long != ab_bl:
            print('Baseline not identical to followup in subject {} of ab long {} and ab bl {}'.format(id, ab_long, ab_bl))
            continue
        # Set binary_label based on conditions

        if baseline:
            ab_long = ab_bl
        if num_classes == 2:
            if diagnosis in class_pairs[0] and ab_long == 0:
                binary_label = 0 
            elif diagnosis in class_pairs[1] and ab_long == 1:
                binary_label = 1
            else:
                continue
        else:
            if diagnosis in class_pairs[0] and ab_long == 0:
                binary_label = 0
            elif  diagnosis in class_pairs[0] and ab_long == 1:
                binary_label = 1
            elif diagnosis in class_pairs[1] and ab_long == 1:
                binary_label = 2   
            elif diagnosis in class_pairs[2] and ab_long == 1:
                binary_label = 3
            else:
                continue
            
        # Process the subject data
        data = process_subject(id, float(binary_label), mat_files_dir, percentile=percentile, connectome=connectome)
    
        # Append the processed data if not None
        if data is not None:
            all_data.append(data)
    return all_data
mat_files_dir = "/media/hang/EXTERNAL_US/Data/1_HANG_FDG_PET/ADNI_Second_organized/KDE_Results_corrected_by_age_sec_education/"
num_classes = 2
connectome = 'Z_scoring'
class_pairs = [['CN', 'SMC'], ['MCI', 'EMCI', 'LMCI']]
data = read_data( class_pairs, mat_files_dir,connectome, num_classes,percentile=10)
print(len(data))
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
