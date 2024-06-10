import pandas as pd
import numpy as np
import os
import timeit
import networkx as nx
import os
import numpy as np
import torch
import networkx as nx
from scipy.io import loadmat
from torch_geometric.data import Data
from torch_geometric.utils import  remove_self_loops
import multiprocessing
import glob
from sklearn.utils import resample
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
        edge_att[i] = edge_features[adj.row[i], adj.col[i]]
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



def read_data( class_pairs, mat_files_dir,connectome, num_classes,baseline=True,percentile=10,resample_data=True):
    all_data = []
    ADNI_merge = pd.read_csv('/media/hang/EXTERNAL_US/Data/1_HANG_FDG_PET/ADNIMERGE_30Jan2024.csv')
    UCBERKLEY_PET = pd.read_excel('/home/hang/GU/Project/AD_classification_synthesis/data/ADNI/UCBERKELEYAV45_01_14_21.xlsx')


    # Path to the raw data
    data_list_included = glob.glob(os.path.join(mat_files_dir,'*_Z_scoring.mat'))

    # Extract subject IDs from file names
    subject_ids_included = [filename.split('/')[-1].split('_Z_scoring.mat')[0] for filename in data_list_included]


    binary_label_counts = {}
    subject_list = []
    for id in subject_ids_included:
        # Get the diagnosis values sorted by exam date for the subject
        diagnosis_values = ADNI_merge[ADNI_merge['PTID'] == id].sort_values('EXAMDATE')['DX'].dropna().values
        
        # Check if there is more than one unique diagnosis
        if len(set(diagnosis_values)) > 1:
            continue

        # Extract RID from the subject ID
        rid = id.split('_')[-1]
        #print(f"Extracted RID: {rid}")

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
                subject_list.append(id)
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
        data = process_subject(id, binary_label, mat_files_dir, percentile=percentile, connectome=connectome)
    
        # Append the processed data if not None
        if data is not None:
            all_data.append(data)
            binary_label_counts[binary_label] = binary_label_counts.get(binary_label, 0) + 1

    # Resample the data if requested
    if resample_data:
            # Group the data by binary label
        grouped_data = {label: [] for label in binary_label_counts.keys()}
        for data in all_data:
            grouped_data[int(data.y.item())].append(data)
        
        # Find the minimum class count
        min_class_count = min(binary_label_counts.values())
        
        # Resample each group to the size of the smallest group
        resampled_data = []
        for label, data_list in grouped_data.items():
            if len(data_list) > min_class_count:
                data_list = resample(data_list, replace=False, n_samples=min_class_count, random_state=42)
            resampled_data.extend(data_list)
        
        all_data = resampled_data
        binary_label_counts = {label: min_class_count for label in binary_label_counts.keys()}

            # Print out the counts of unique values in binary_label after resampling
    print("Unique value counts in binary labels after resampling:", binary_label_counts)
    print(subject_list)
    return all_data
def get_connectome_path_and_label(connectome_path, class_pair):
    class_map = {
        'CN': ('CNnegLong_connectome.mat', 0),
        'CNpos': ('CNposSUVRValues_connectome.mat', 1),
        'MCI': ('MCISUVRValues_connectome.mat', 1),
        'Dementia': ('ADSUVRValues_connectome.mat', 1)
    }

    class_name_0, class_name_1 = class_pair

    connectome_paths_labels = []

    if class_name_0 in class_map:
        if class_name_0 == 'CN':


            new_last_substring = 'fp_CN_as_reference'
            delimiter = '/'

            connectome_path = switch_last_substring(connectome_path, delimiter, new_last_substring)
        if class_name_0 == 'CN' and class_name_1 == 'CN':
            connectome_paths_labels.append((os.path.join(connectome_path, class_map['CN'][0]), 0))
            connectome_paths_labels.append((os.path.join(connectome_path, class_map['CNpos'][0]), 1))
        else:
            mat_path_0, binary_label_0 = class_map[class_name_0]
            connectome_paths_labels.append((os.path.join(connectome_path, mat_path_0), 0))

    if class_name_1 in class_map and not (class_name_0 == 'CN' and class_name_1 == 'CN'):
        mat_path_1, binary_label_1 = class_map[class_name_1]
        connectome_paths_labels.append((os.path.join(connectome_path, mat_path_1), binary_label_1))

    return connectome_paths_labels

def load_for_perturbation(baseline, class_pair, percentile,resample_data):
    input_file_path = '/media/hang/EXTERNAL_US/Data/1_HANG_FDG_PET/IndividualConnectome'
    all_data = []


    # Determine the correct connectome path based on baseline
    connectome_path = os.path.join(input_file_path, 'bl_CN_as_reference') if baseline else os.path.join(input_file_path, 'fp_CN_as_reference')

    # Get the connectome paths and binary labels
    connectome_info = get_connectome_path_and_label(connectome_path, class_pair)

    if not connectome_info:
        print("No valid class pair found.")
        return all_data

    for connectome, binary_label in connectome_info:
        print(connectome,binary_label)
        print(f"Processing connectome: {connectome} with label: {binary_label}")

        if not os.path.exists(connectome):
            print(f"Connectome file not found: {connectome}")
            continue

        brain_connectome = loadmat(connectome)

        node_features_list = None
        edge_features_list = None

        for keys, value in brain_connectome.items():
            if 'SUVRValues' in keys or 'IndividualAmyloidConnectome' in keys:
                node_features_list = value
                print(f"Node feature list loaded: {keys}")
            if 'partial_corrs' in keys:
                edge_features_list = value
                print(f"Edge feature list loaded: {keys}")
        
        if node_features_list is None or edge_features_list is None:
            print(f"Skipping {connectome} due to missing data.")
            continue

        for idx in range(len(node_features_list)):
            node_features = node_features_list[idx]
            num_nodes = node_features.shape[0]
            edge_features = edge_features_list[idx]

            node_features = apply_threshold(node_features, percentile=percentile)
            edge_features = apply_threshold(edge_features, percentile=percentile)
            
            G = nx.from_numpy_array(edge_features)
            A = nx.to_scipy_sparse_array(G, format='coo')
            adj = A.tocoo()
            edge_att = np.zeros(len(adj.row))
            for i in range(len(adj.row)):
                edge_att[i] = edge_features[adj.row[i], adj.col[i]]
            edge_index = np.stack([A.row, A.col])
            edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
            edge_index = edge_index.long()
            edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes, num_nodes)
            pos = torch.eye(num_nodes)  # Using an identity matrix for positional data

            all_data.append(Data(x=torch.tensor(node_features, dtype=torch.float),
                                 edge_index=edge_index,
                                 edge_attr=edge_att,
                                 y=torch.tensor([binary_label], dtype=torch.long),
                                 pos=pos))
        # Resample the data if requested
    if resample_data:
        # Group the data by binary label
        binary_label_counts = {0: 0, 1: 0}
        for data in all_data:
            binary_label_counts[int(data.y.item())] += 1

        grouped_data = {label: [] for label in binary_label_counts.keys()}
        for data in all_data:
            grouped_data[int(data.y.item())].append(data)
        
        # Find the minimum class count
        min_class_count = min(binary_label_counts.values())
        
        # Resample each group to the size of the smallest group
        resampled_data = []
        for label, data_list in grouped_data.items():
            if len(data_list) > min_class_count:
                data_list = resample(data_list, replace=False, n_samples=min_class_count, random_state=42)
            resampled_data.extend(data_list)
        
        all_data = resampled_data

        # Print out the counts of unique values in binary_label after resampling
        print("Unique value counts in binary labels after resampling:", {label: min_class_count for label in binary_label_counts.keys()})


    
    return all_data
def switch_last_substring(s, delimiter, new_last_substring):
    parts = s.split(delimiter)
    if len(parts) < 2:
        return s  # Return the original string if there is no delimiter
    
    parts[-1] = new_last_substring
    return delimiter.join(parts)

# # Example usage
# class_pairs = [
#     ('CN', 'MCI'),
#     # ('CN', 'Dementia'),
#     # ('CN', 'CN'),  # Assuming 'CN ab+' is represented like this in the 'DX_bl' column
#     # ('MCI', 'Dementia')
# ]

# for class_pair in class_pairs:
#     all_data = load_for_perturbation(baseline=True, class_pair=class_pair)
#     print(f"Class pair {class_pair}: {len(all_data)} data entries loaded.")



# mat_files_dir = "/media/hang/EXTERNAL_US/Data/1_HANG_FDG_PET/ADNI_Second_organized/KDE_Results_corrected_by_age_sec_education"
# # mat_files_dir = "/media/hang/EXTERNAL_US/Data/1_HANG_FDG_PET/longitudinal_AD_MCI_CN/MIXED_ALL_AGE_SEX_EDU_CORRECTED"
# num_classes = 2
# connectome = 'Z_scoring'
# class_pairs = [
#      (['CN', 'SMC'], ['MCI', 'EMCI', 'LMCI']),
# #    (['CN', 'SMC'], 'Dementia'),
# #     (['CN', 'SMC'], ['CN', 'SMC']),  # Assuming 'CN ab+' is represented like this in the 'DX_bl' column
# #(['CN', 'SMC'], ['CN', 'SMC'],['EMCI', 'LMCI'],'AD')
# ]
# for class_pair in class_pairs:
#     print(class_pair)
#     data = read_data( class_pair, mat_files_dir,connectome, num_classes,percentile=10,resample_data=False)
# # print(len(data))
# class_pairs = [
#     (['CN'], ['MCI', 'AD']),  # Example class pair
#     # Add more class pairs as needed
# ]


# # Now, `all_data` contains a list of Data objects ready for GNN processing.
