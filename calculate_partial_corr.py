import scipy
import os
import numpy as np
from nilearn import connectome
from scipy.io import loadmat, savemat  # Corrected imports
import glob

def extract_second_elements_from_mat(file_path):
    """
    Extracts the second element of each 1x2 cell array from a (120,1) cell array stored in a .mat file.
    
    Parameters:
    - file_path: Path to the .mat file.
    
    Returns:
    - A NumPy array of shape (500, 120), each representing the second element from each 1x2 cell array.
    """
    mat_contents = loadmat(file_path)
    kde_results = mat_contents['kde_results']
    corrected_arrays = []
    
    for i in range(kde_results.shape[0]):
        second_element = kde_results[i, 0][0, 1]
        if second_element.ndim == 1:
            second_element = second_element[:, np.newaxis]

        corrected_arrays.append(second_element)
    
    final_array = np.hstack(corrected_arrays)
    return final_array

def get_KDE(root_dir, mat_files_dir):
    conn_measure = connectome.ConnectivityMeasure(kind='partial correlation')
    subj_kde_list = [extract_second_elements_from_mat(kde_path) for kde_path in mat_files_dir]
    connectivity = conn_measure.fit_transform(subj_kde_list)
    for i, subj_kde in enumerate(subj_kde_list):

        # For partial correlation, we treat each (500, 120) as a single subject's data

        subj_ID = mat_files_dir[i].split('/')[-1].split('_KDE')[0]
        
        output_path = os.path.join(root_dir, f"{subj_ID}_partial_correlation_KDE.mat")
        savemat(output_path, {'partial_corr': connectivity[i]})

root_dir = ['/media/hang/EXTERNAL_US/Data/1_HANG_FDG_PET/longitudinal_AD_MCI_CN/6_Hang_CN_neg/connectome',
            '/media/hang/EXTERNAL_US/Data/1_HANG_FDG_PET/longitudinal_AD_MCI_CN/6_Hang_MCI_pos/connectome',
            '/media/hang/EXTERNAL_US/Data/1_HANG_FDG_PET/longitudinal_AD_MCI_CN/6_HANG_CN_pos/connectome',
           '/media/hang/EXTERNAL_US/Data/1_HANG_FDG_PET/longitudinal_AD_MCI_CN/6_HANG_AD_pos/connectome']
for path in root_dir:
    mat_files_dir = glob.glob(os.path.join(path, '*KDE*'))
    get_KDE(path, mat_files_dir)
