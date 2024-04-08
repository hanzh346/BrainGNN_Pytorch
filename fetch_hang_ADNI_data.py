import pandas as pd
import numpy as np
import scipy.io
import os
# Assume dataTable.csv is your dataset containing subject IDs, diagnosis info, etc.
dataTable = pd.read_csv('/home/hang/GitHub/BrainGNN_Pytorch/data/filtered_selectedDataUnique_merged_ADNI.csv')
mat_files_dir = "/media/hang/EXTERNAL_US/Data/1_HANG_FDG_PET/ADNI_Second_organized/KDE_Results/" 
graph_measure_path = '/media/hang/EXTERNAL_US/Data/1_HANG_FDG_PET/ADNI_Second_organized/organized/KDE_Results/reorgnized_AllMeasuresAndDiagnosisByThreshold_DISTANCE.mat'
def load_graph_measures(file_path):
    # Adjusted for the new structure name
    data = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    combinedData = data['newData']  

    all_subjects_measures = {}

    for entry in combinedData:
        subject_id = entry.SubjectID
        diagnosis = entry.Diagnosis
        measures_struct = entry.Measures
        
        current_subject_measures = {'Diagnosis': diagnosis}
        
        for measure_name in ['bc', 'Eglobal', 'Elocal', 'Cp', 'Lp', 'sigma']:
            measure_value = getattr(measures_struct, measure_name, None)
            if measure_value is not None:
                current_subject_measures[measure_name] = np.array(measure_value).flatten()
            else:
                current_subject_measures[measure_name] = None

        all_subjects_measures[subject_id] = current_subject_measures

    return all_subjects_measures

def extract_features(mat_file_path):
    data = scipy.io.loadmat(mat_file_path)
    features = data['scaledMahalDistMatrix']
    return features

def apply_threshold(features, percentile=20):
    threshold = np.percentile(features, 100 - percentile)
    return np.where(features > threshold, features, 0)

def prepare_graph_measures(all_subjects_measures, PTID):
    subject_measures = all_subjects_measures.get(PTID, None)
    if not subject_measures:
        return None
    measures_array = np.array([subject_measures[measure] for measure in ['bc', 'Eglobal', 'Elocal', 'Cp', 'Lp', 'sigma'] if subject_measures[measure] is not None]).flatten()
    return measures_array

def perform_classification(dataTable, class_pairs, mat_files_dir, graph_measure_path):
    all_subjects_measures = load_graph_measures(graph_measure_path)
    for class_0_labels, class_1_labels in class_pairs:
        # Ensure class_1_labels is always a list for consistent processing
        if not isinstance(class_1_labels, list):
            class_1_labels = [class_1_labels]
        
        class_1_label_str = "_vs_".join(class_1_labels)  # For use in file naming
        
        dataTable['binary_labels'] = dataTable.apply(lambda row:
            0 if row['DX_bl'] in class_0_labels and row['AV45'] < 1.11 else
            1 if row['DX_bl'] in class_1_labels and row['AV45'] >= 1.11 else np.nan, axis=1)
        filtered_data = dataTable.dropna(subset=['binary_labels'])
        percentile = 20

        
        X_features, X_measures, y = [], [], []
        for _, row in filtered_data.iterrows():
            PTID = row['PTID']
            mat_file_path = os.path.join(mat_files_dir, f"{PTID}_ScaledMahalanobisDistanceMatrix.mat")
            if os.path.exists(mat_file_path):
                features = extract_features(mat_file_path)
                features = apply_threshold(features, percentile=percentile)
                
                measures_array = prepare_graph_measures(all_subjects_measures, PTID)
                
                if measures_array is not None and features.size > 0:
                    X_features.append(features)
                    X_measures.append(measures_array)
                    y.append(row['binary_labels'])
    
    # Convert lists to NumPy arrays for machine learning processing
    X_features = np.array(X_features)
    X_measures = np.array(X_measures)
    y = np.array(y)
    
    # Further processing and classification can follow here
    # This might include concatenating features and measures, splitting data, training, and testing the model
    
    return X_features, X_measures, y
class_pairs = [
     (['CN', 'SMC'], ['EMCI', 'LMCI']),
     (['CN', 'SMC'], 'AD'),
     (['CN', 'SMC'], ['CN', 'SMC']),  # Assuming 'CN ab+' is represented like this in the 'DX_bl' column
    # (['EMCI', 'LMCI'],'AD'),
]