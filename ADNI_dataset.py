
import os
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from scipy.io import loadmat
import numpy as np
import networkx as nx
from torch_geometric.data import DataLoader
import multiprocessing
from functools import partial
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from os import listdir
import os.path as osp
from fetch_hang_ADNI_data import *

# class ADNI_DATASET(InMemoryDataset):
#     def __init__(self, root, csv_file, transform=None, pre_transform=None):
#         """
#         Initializes the dataset.

#         Parameters:
#             root (str): Directory where the .mat files are stored.
#             csv_file (str): Full path to the CSV file with metadata.
#             transform (callable, optional): Optional transform to be applied on a sample.
#             pre_transform (callable, optional): Optional transform to be applied before saving the data to disk.
#         """
#         self.csv_file = csv_file
#         super(ADNI_DATASET, self).__init__(root, transform=transform, pre_transform=pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_file_names(self):
#         # We don't use raw file names because we process everything directly from the CSV and .mat files.
#         return []

#     @property
#     def processed_file_names(self):
#         # Names of the processed files you expect to save and load.
#         return ['data.pt']

#     def download(self):
#         # Implement this method if you need to automatically download the data.
#         pass

#     def process(self):
#         # Reads the data, processes it, and saves it in the 'processed' directory.
#         dataTable = pd.read_csv(self.csv_file)
#         class_pairs = [
#             (['CN', 'SMC'], ['EMCI', 'LMCI']),
#         ]

#         self.data,self.slices = read_data(dataTable, class_pairs, self.root)  # Ensure read_data correctly processes and returns a list of Data objects
#  # `collate` correctly combines Data objects
#         os.makedirs(self.processed_dir, exist_ok=True)
#         torch.save((self.data, self.slices), self.processed_paths[0])
        
#     def __len__(self):
#         return len(self.data.y) if isinstance(self.data, list) else 0  # Ensure self.data is a list and return its length

#     def get(self, idx):
#         # Support indexing to get a specific graph
#         return self.data[idx]
#     def __repr__(self):
#         data_info = f"{len(self.data.y)}"
#         return f"{self.__class__.__name__}({data_info})"
class ADNI_DATASET(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        super(ADNI_DATASET, self).__init__(root,transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        data_dir = osp.join(self.root,'raw')
        onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        onlyfiles.sort()
        return onlyfiles
    @property
    def processed_file_names(self):
        return  'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        return

    def process(self):

        dataTable = pd.read_csv(self.name)
        class_pairs = [
            (['CN', 'SMC'], ['EMCI', 'LMCI']),
         ]

        self.data,self.slices = read_data(dataTable, class_pairs, self.raw_dir)  # Ensure read_data correctly processes and returns a list of Data objects

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.root, len(self))

mat_files_dir = "/media/hang/EXTERNAL_US/Data/1_HANG_FDG_PET/ADNI_Second_organized/KDE_Results/" 
dataset = ADNI_DATASET(root=mat_files_dir, name='/home/hang/GitHub/BrainGNN_Pytorch/data/filtered_selectedDataUnique_merged_ADNI.csv')
print(dataset)