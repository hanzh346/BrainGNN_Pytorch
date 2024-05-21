
import os
import pandas as pd
import torch
from os import listdir
import numpy as np
import os.path as osp
from fetch_hang_ADNI_data import read_data, process_subject
from torch_geometric.data import InMemoryDataset, Data, Batch


class ADNI_DATASET(InMemoryDataset):
    def __init__(self, root, csv_file, transform=None, pre_transform=None):
        self.root = root
        self.csv_file = csv_file  # Renamed for clarity
        super(ADNI_DATASET, self).__init__(root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0]) if osp.exists(self.processed_paths[0]) else (None, None)

    @property
    def raw_file_names(self):
        data_dir = osp.join(self.root,'raw')
        onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
        onlyfiles.sort()
        return onlyfiles

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Method to download the dataset, if it's not already available locally
        pass

    def process(self):
        dataTable = pd.read_csv(self.csv_file)
        class_pairs = [('CN', 'SMC'), ('EMCI', 'LMCI')]
        self.data, batch = read_data(dataTable, class_pairs, osp.join(self.root, 'raw'))
        self.data, self.slices = self.collate(Batch.from_data_list( self.data))
        torch.save((self.data, self.slices), self.processed_paths[0])

    def split(self,data, batch):
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


    def __repr__(self):
        return '{}({})'.format(self.root, len(self))
    



    # def __getitem__(self, idx):
    #         # If the index is an integer, retrieve a single example
    #         if isinstance(idx, int):
    #             return self.get(idx)
            
    #         # If the index is a slice or list, retrieve a subset
    #         elif isinstance(idx, (slice, list)):
    #             # The collate function can take a list of Data objects and merge them into a single batch
    #             return self.collate([self.get(i) for i in idx])
    # def get(self, idx):
    #         # Assuming that self.data and self.slices are properly set up to return
    #         # a Data object corresponding to the graph at the given index
    #         # We use self.slices to get the correct data range for the given index
    #         data_idx = slice(self.slices['x'][idx], self.slices['x'][idx + 1])
    #         return Data(x=self.data.x[data_idx],
    #                     edge_index=self.data.edge_index[:, data_idx],
    #                     edge_attr=self.data.edge_attr[data_idx],
    #                     y=self.data.y[idx])



