from fetch_hang_ADNI_data import *
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split, KFold
from torch_geometric.data import Data
import torch
import os
import numpy as np
import argparse
import time
import copy
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from imports.ABIDEDataset import ABIDEDataset
from net.braingnn import Network
from imports.utils import train_val_test_split
from sklearn.metrics import classification_report, confusion_matrix
import networkx as nx
print(nx.__version__)
torch.manual_seed(123)

EPS = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=800, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=400, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='home/hang/GitHub/BrainGNN_Pytorch/data/ABIDE_pcp/cpac/filt_noglobal', help='root directory of the dataset')
parser.add_argument('--fold', type=int, default=0, help='training which fold')
parser.add_argument('--lr', type = float, default=0.01, help='learning rate')
parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
parser.add_argument('--weightdecay', type=float, default=5e-3, help='regularization')
parser.add_argument('--lamb0', type=float, default=1, help='classification loss weight')
parser.add_argument('--lamb1', type=float, default=0, help='s1 unit regularization')
parser.add_argument('--lamb2', type=float, default=0, help='s2 unit regularization')
parser.add_argument('--lamb3', type=float, default=0.1, help='s1 entropy regularization')
parser.add_argument('--lamb4', type=float, default=0.1, help='s2 entropy regularization')
parser.add_argument('--lamb5', type=float, default=0.1, help='s1 consistence regularization')
parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
parser.add_argument('--ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--indim', type=int, default=200, help='feature dim')
parser.add_argument('--nroi', type=int, default=200, help='num of ROIs')
parser.add_argument('--nclass', type=int, default=2, help='num of classes')
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
parser.add_argument('--save_path', type=str, default='./model/', help='path to save model')
opt = parser.parse_args()

if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)
# Assuming dataTable, class_pairs, mat_files_dir, graph_measure_path are defined
# Assume dataTable.csv is your dataset containing subject IDs, diagnosis info, etc.
dataTable = pd.read_csv('/home/hang/GitHub/BrainGNN_Pytorch/data/filtered_selectedDataUnique_merged_ADNI.csv')
mat_files_dir = "/media/hang/EXTERNAL_US/Data/1_HANG_FDG_PET/ADNI_Second_organized/KDE_Results/" 
graph_measure_path = '/media/hang/EXTERNAL_US/Data/1_HANG_FDG_PET/ADNI_Second_organized/organized/KDE_Results/reorgnized_AllMeasuresAndDiagnosisByThreshold_DISTANCE.mat'
class_pairs = [
     (['CN', 'SMC'], ['EMCI', 'LMCI']),
     (['CN', 'SMC'], 'AD'),
     (['CN', 'SMC'], ['CN', 'SMC']),  # Assuming 'CN ab+' is represented like this in the 'DX_bl' column
    # (['EMCI', 'LMCI'],'AD'),
]

path = opt.dataroot
save_model = opt.save_model
load_model = opt.load_model
opt_method = opt.optim
num_epoch = opt.n_epochs
fold = opt.fold
writer = SummaryWriter(os.path.join('./log',str(fold)))


X_features, X_measures, y = perform_classification(dataTable, class_pairs, mat_files_dir, graph_measure_path)
# Convert to PyTorch tensors
X_features_tensor = torch.tensor(X_features, dtype=torch.float)
X_measures_tensor = torch.tensor(X_measures, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.long)
# Concatenate along feature dimension
X_tensor = X_features_tensor#torch.cat((X_features_tensor, X_measures_tensor), dim=1)
# Assuming X_tensor and y_tensor are your dataset tensors
num_features = X_tensor.size(1)
num_classes = 2  # Adjust according to your specific problem

# KFold cross-validator
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Splitting dataset indices for cross-validation
indices = np.arange(len(X_tensor))

class CustomDataset(Dataset):
    """Custom dataset to facilitate loading."""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

############### Define Graph Deep Learning Network ##########################
model = Network(opt.indim,opt.ratio,opt.nclass).to(device)
print(model)

if opt_method == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
elif opt_method == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr =opt.lr, momentum = 0.9, weight_decay=opt.weightdecay, nesterov = True)

scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

############################### Define Other Loss Functions ########################################
def topk_loss(s,ratio):
    if ratio > 0.5:
        ratio = 1-ratio
    s = s.sort(dim=1).values
    res =  -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:int(s.size(1)*ratio)]+EPS).mean()
    return res


def consist_loss(s):
    if len(s) == 0:
        return 0
    s = torch.sigmoid(s)
    W = torch.ones(s.shape[0],s.shape[0])
    D = torch.eye(s.shape[0])*torch.sum(W,dim=1)
    L = D-W
    L = L.to(device)
    res = torch.trace(torch.transpose(s,0,1) @ L @ s)/(s.shape[0]*s.shape[0])
    return res

###################### Network Training Function#####################################
def train(epoch):
    print('train...........')
    

    for param_group in optimizer.param_groups:
        print("LR", param_group['lr'])
    model.train()
    s1_list = []
    s2_list = []
    loss_all = 0
    step = 0
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            # If data is a list, unpack and move each element to the device as needed
            data = [item.to(device) for item in data]
        else:
            # Directly move to device if data is a tensor or a Data object
            data = data.to(device)

        optimizer.zero_grad()
        output, w1, w2, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
        s1_list.append(s1.view(-1).detach().cpu().numpy())
        s2_list.append(s2.view(-1).detach().cpu().numpy())

        loss_c = F.nll_loss(output, data.y)

        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = topk_loss(s1,opt.ratio)
        loss_tpk2 = topk_loss(s2,opt.ratio)
        loss_consist = 0
        for c in range(opt.nclass):
            loss_consist += consist_loss(s1[data.y == c])
        loss = opt.lamb0*loss_c + opt.lamb1 * loss_p1 + opt.lamb2 * loss_p2 \
                   + opt.lamb3 * loss_tpk1 + opt.lamb4 *loss_tpk2 + opt.lamb5* loss_consist
        writer.add_scalar('train/classification_loss', loss_c, epoch*len(train_loader)+step)
        writer.add_scalar('train/unit_loss1', loss_p1, epoch*len(train_loader)+step)
        writer.add_scalar('train/unit_loss2', loss_p2, epoch*len(train_loader)+step)
        writer.add_scalar('train/TopK_loss1', loss_tpk1, epoch*len(train_loader)+step)
        writer.add_scalar('train/TopK_loss2', loss_tpk2, epoch*len(train_loader)+step)
        writer.add_scalar('train/GCL_loss', loss_consist, epoch*len(train_loader)+step)
        step = step + 1

        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

        s1_arr = np.hstack(s1_list)
        s2_arr = np.hstack(s2_list)
    scheduler.step()
    return loss_all / len(train_dataset), s1_arr, s2_arr ,w1,w2


###################### Network Testing Function#####################################
def test_acc(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        outputs= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
        pred = outputs[0].max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()

    return correct / len(loader.dataset)

def test_loss(loader,epoch):
    print('testing...........')
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        output, w1, w2, s1, s2= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
        loss_c = F.nll_loss(output, data.y)

        loss_p1 = (torch.norm(w1, p=2)-1) ** 2
        loss_p2 = (torch.norm(w2, p=2)-1) ** 2
        loss_tpk1 = topk_loss(s1,opt.ratio)
        loss_tpk2 = topk_loss(s2,opt.ratio)
        loss_consist = 0
        for c in range(opt.nclass):
            loss_consist += consist_loss(s1[data.y == c])
        loss = opt.lamb0*loss_c + opt.lamb1 * loss_p1 + opt.lamb2 * loss_p2 \
                   + opt.lamb3 * loss_tpk1 + opt.lamb4 *loss_tpk2 + opt.lamb5* loss_consist

        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)

#######################################################################################
############################   Model Training #########################################
#######################################################################################
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 1e10
for fold, (train_ids, test_ids) in enumerate(kfold.split(indices)):
    print(f'Fold {fold+1}/{k_folds}')
    
    # Further split train_ids into training and validation sets
    train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)
    
    # Extracting datasets for the current fold
    X_train, y_train = X_tensor[train_ids], y_tensor[train_ids]
    X_val, y_val = X_tensor[val_ids], y_tensor[val_ids]
    X_test, y_test = X_tensor[test_ids], y_tensor[test_ids]
    
    # Convert to Dataset and DataLoader
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    test_dataset = CustomDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    for epoch in range(0, num_epoch):
        since  = time.time()
        tr_loss, s1_arr, s2_arr, w1, w2 = train(epoch)
        tr_acc = test_acc(train_loader)
        val_acc = test_acc(val_loader)
        val_loss = test_loss(val_loader,epoch)
        time_elapsed = time.time() - since
        print('*====**')
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Epoch: {:03d}, Train Loss: {:.7f}, '
            'Train Acc: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}'.format(epoch, tr_loss,
                                                        tr_acc, val_loss, val_acc))

        writer.add_scalars('Acc',{'train_acc':tr_acc,'val_acc':val_acc},  epoch)
        writer.add_scalars('Loss', {'train_loss': tr_loss, 'val_loss': val_loss},  epoch)
        writer.add_histogram('Hist/hist_s1', s1_arr, epoch)
        writer.add_histogram('Hist/hist_s2', s2_arr, epoch)

        if val_loss < best_loss and epoch > 5:
            print("saving best model")
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            if save_model:
                torch.save(best_model_wts, os.path.join(opt.save_path,str(fold)+'.pth'))

#######################################################################################
######################### Testing on testing set ######################################
#######################################################################################

    if opt.load_model:
        model = Network(opt.indim,opt.ratio,opt.nclass).to(device)
        model.load_state_dict(torch.load(os.path.join(opt.save_path,str(fold)+'.pth')))
        model.eval()
        preds = []
        correct = 0
        for data in val_loader:
            data = data.to(device)
            outputs= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
            pred = outputs[0].max(1)[1]
            preds.append(pred.cpu().detach().numpy())
            correct += pred.eq(data.y).sum().item()
        preds = np.concatenate(preds,axis=0)
        trues = val_dataset.data.y.cpu().detach().numpy()
        cm = confusion_matrix(trues,preds)
        print("Confusion matrix")
        print(classification_report(trues, preds))

    else:
        model.load_state_dict(best_model_wts)
        model.eval()
        test_accuracy = test_acc(test_loader)
        test_l= test_loss(test_loader,0)
        print("===========================")
        print("Test Acc: {:.7f}, Test Loss: {:.7f} ".format(test_accuracy, test_l))
        print(opt)