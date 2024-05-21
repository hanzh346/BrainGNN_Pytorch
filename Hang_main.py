from fetch_hang_ADNI_data import *
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
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
from torch_geometric.data import DataLoader
from sklearn.preprocessing import label_binarize
from  ADNI_dataset import ADNI_DATASET
from net.braingnn import Network
from imports.utils import train_val_test_split
from sklearn.metrics import classification_report, confusion_matrix
import networkx as nx
import torch_geometric
import json
import os.path as osp
from datetime import datetime
print(nx.__version__)
torch.manual_seed(123)

EPS = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_path = '/home/hang/GitHub/BrainGNN_Pytorch/results'
folder_path = os.path.join(base_path, current_time)


percentiles = [10, 20, 50, 100]
# Assuming dataTable, class_pair, mat_files_dir, graph_measure_path are defined
# Assume dataTable.csv is your dataset containing subject IDs, diagnosis info, etc.
#dataTable = pd.read_csv('/home/hang/GitHub/BrainGNN_Pytorch/data/filtered_selectedDataUnique_merged_ADNI.csv')
mat_files_dir = "/media/hang/EXTERNAL_US/Data/1_HANG_FDG_PET/ADNI_Second_organized/KDE_Results_corrected_by_age_sec_education/" 
graph_measure_path = '/media/hang/EXTERNAL_US/Data/1_HANG_FDG_PET/ADNI_Second_organized/organized/KDE_Results/reorgnized_AllMeasuresAndDiagnosisByThreshold_DISTANCE.mat'
dataTable = pd.read_csv('/media/hang/EXTERNAL_US/Data/1_HANG_FDG_PET/ADNI_Second_organized/merged_filtered_selected_data_ADNI_merge_with_UCBERKELEYAV45_01_14_21.csv')

class_pairs = [
     (['CN', 'SMC'], ['MCI', 'EMCI', 'LMCI']),
    (['CN', 'SMC'], 'Dementia'),
     (['CN', 'SMC'], ['CN', 'SMC']),  # Assuming 'CN ab+' is represented like this in the 'DX_bl' column
#(['CN', 'SMC'], ['CN', 'SMC'],['EMCI', 'LMCI'],'AD')
]
#connectomes = ['Z_scoring']
connectomes = ['ScaledMahalanobisDistanceMatrix', 'Z_scoring', 'K_correlation', 'K_JS_Divergence']
#connectomes = ['Z_scoringLongitude']
for connectome in connectomes:
    print(connectome)
    for percentile in percentiles:
        for class_pair in class_pairs:
            print(class_pair)
            parser = argparse.ArgumentParser()
            parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
            parser.add_argument('--n_epochs', type=int, default=50 , help='number of epochs of training')
            parser.add_argument('--batchSize', type=int, default=32, help='size of the batches')
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
            parser.add_argument('--indim', type=int, default=120, help='feature dim')
            parser.add_argument('--nroi', type=int, default=120, help='num of ROIs')
            parser.add_argument('--nclass', type=int, default=2, help='num of classes')
            parser.add_argument('--baseline', type=bool,default = True)
            parser.add_argument('--load_model', type=bool, default=False)
            parser.add_argument('--save_model', type=bool, default=True)
            parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
            parser.add_argument('--save_path_model', type=str, default=osp.join(folder_path,connectome,str(percentile),f"{class_pair[0]}_vs_{class_pair[1]}",'model/'), help='path to save model')
            parser.add_argument('--save_path_results', type=str, default=osp.join(folder_path,connectome,str(percentile),f"{class_pair[0]}_vs_{class_pair[1]}",'results/'), help='path to save results')
            opt = parser.parse_args()
            if opt.load_model:
                date = '2024-04-16_21-46-58'
                opt.save_path_model = os.path.join(base_path,date,osp.join(connectome,str(percentile),f"{class_pair[0]}_vs_{class_pair[1]}",'model/'))
                opt.save_path_results = os.path.join(base_path,date,osp.join(connectome,str(percentile),f"{class_pair[0]}_vs_{class_pair[1]}",'results/'))


            if not os.path.exists(opt.save_path_model) and not opt.load_model:
                os.makedirs(opt.save_path_model)
            if not os.path.exists(opt.save_path_results) and not opt.load_model:
                os.makedirs(opt.save_path_results)

            path = opt.dataroot
            save_model = opt.save_model
            load_model = opt.load_model
            opt_method = opt.optim
            num_epoch = opt.n_epochs
            fold = opt.fold
            writer = SummaryWriter(os.path.join('./log',str(fold)))

            # Adjust read_data to return a list of Data objects and corresponding labels
            # data_list = ADNI_DATASET(root=mat_files_dir, csv_file='/home/hang/GitHub/BrainGNN_Pytorch/data/filtered_selectedDataUnique_merged_ADNI.csv')
            #data_list = read_data(dataTable, class_pair, osp.join(mat_files_dir, 'raw'),connectome,percentile=percentile,num_classes=opt.nclass)
            data_list = read_data(class_pair, mat_files_dir,connectome,percentile=percentile,num_classes=opt.nclass, baseline=opt.baseline)  

            # KFold cross-validator
            k_folds = 5
            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            # Splitting data_list indices for cross-validation
            indices = np.arange(len(data_list))
            # data_list.data.y = data_list.data.y.squeeze()
            # data_list.data.x[data_list.data.x == float('inf')] = 0


            ############### Define Graph Deep Learning Network ##########################
            model = Network(opt.indim,opt.ratio,opt.nclass).to(device)

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
                model.train().float()
                s1_list = []
                s2_list = []
                loss_all = 0
                step = 0
                if not train_loader:
                    raise ValueError("Train loader is empty. Check your dataset and splits.")

                for data in train_loader:
                    if not data:
                        continue  # Skip any empty batches or misloaded data points
                    if isinstance(data, list):
                        # If data is a list, unpack and move each element to the device as needed
                        data = [item.to(device) for item in data]
                    else:
                        # Directly move to device if data is a tensor or a Data object
                        data = data.to(device)


                    optimizer.zero_grad()
                    output, w1, w2, s1, s2, perm1, perm2 = model(data.x.float(), data.edge_index, data.batch, data.edge_attr.float(), data.pos)
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
                    outputs= model(data.x.float(), data.edge_index, data.batch, data.edge_attr.float(),data.pos)
                    pred = outputs[0].max(dim=1)[1]
                    correct += pred.eq(data.y).sum().item()

                return correct / len(loader.dataset)

            def test_loss(loader,epoch):
                print('testing...........')
                model.eval()
                loss_all = 0
                for data in loader:
                    data = data.to(device)
                    output, w1, w2, s1, s2, perm1, perm2= model(data.x.float(), data.edge_index, data.batch, data.edge_attr.float(),data.pos)
                    print(perm1.shape)
                    print(perm2.shape)

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
            results = {'Accuracy': [], 'ROC AUC': [], 'Specificity': [], 'Sensitivity': []}
            interesting_indices = {}
            class_specific_results = []
            # Assuming you're working within a cross-validation setup
            for fold, (train_idx, test_idx) in enumerate(kfold.split(data_list)):
                # Using the indices to split the dataset
                train_dataset = [data_list[i] for i in train_idx]

                test_dataset = [data_list[i] for i in test_idx]

                # # Splitting the training data into training and validation sets
                # train_idx, val_idx = train_test_split(list(train_idx), test_size=0.2, random_state=42)
                # train_dataset = [data_list[i] for i in train_idx]
                # val_dataset = [data_list[i] for i in val_idx]

                # Create DataLoaders for train, validation, and test
                train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=False,  drop_last=True)
                # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)
                if not opt.load_model:
                    for epoch in range(0, num_epoch):
                        since  = time.time()
                        tr_loss, s1_arr, s2_arr, w1, w2 = train(epoch)
                        tr_acc = test_acc(train_loader)
                        val_acc = test_acc(test_loader)
                        val_loss = test_loss(test_loader,epoch)
                        time_elapsed = time.time() - since
                        print('*====**')
                        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                        print('classpair {} fold {} Epoch: {:03d}, Train Loss: {:.7f}, '
                            'Train Acc: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}'.format(class_pair, fold, epoch, tr_loss,
                                                                        tr_acc, val_loss, val_acc))

                        writer.add_scalars('Acc fold, classpair{}'.format(fold,class_pair),{'train_acc':tr_acc,'val_acc':val_acc},  epoch)
                        writer.add_scalars('Loss fold, classpair{}'.format(fold,class_pair), {'train_loss': tr_loss, 'val_loss': val_loss},  epoch)
                        writer.add_histogram('Hist/hist_s1 fold, classpair{}'.format(fold,class_pair), s1_arr, epoch)
                        writer.add_histogram('Hist/hist_s2 fold, classpair{}'.format(fold,class_pair), s2_arr, epoch)

                        if val_loss < best_loss and epoch > 5:
                            print("saving best model")
                            best_loss = val_loss
                            best_model_wts = copy.deepcopy(model.state_dict())
                            if save_model:
                                torch.save(best_model_wts, os.path.join(opt.save_path_model,str(fold)+'.pth'))

            #######################################################################################
            ######################### Testing on testing set ######################################
            #######################################################################################

                if opt.load_model:
                    model = Network(opt.indim,opt.ratio,opt.nclass).to(device)
                    if os.path.exists(os.path.join(opt.save_path_model,str(fold)+'.pth')):
                        curr_fold = fold
                    else:
                        fold = curr_fold
                    model.load_state_dict(torch.load(os.path.join(opt.save_path_model,str(fold)+'.pth')))
                    model.eval()
                    preds = []
                    prabas = []
                    trues = []
                    weight_n1s = []
                    weight_n2s = []
                    perm2s = []
                    score2s = []
                    correct = 0
                    for idx, data in enumerate(test_loader):
                        data = data.to(device)
                        outputs= model(data.x.float(), data.edge_index, data.batch, data.edge_attr.float(),data.pos)
                        weight_n1 = model.n1[0].weight.data
                        weight_n2 = model.n2[0].weight.data
                        pred = outputs[0].max(1)[1]
                        praba = outputs[0].exp()[:, 1] 

                        weight_n1s.append(weight_n1.cpu().detach().numpy())
                        weight_n2s.append(weight_n2.cpu().detach().numpy())
                        prabas.append(praba.cpu().detach().numpy())
                        preds.append(pred.cpu().detach().numpy())
                        correct += pred.eq(data.y).sum().item()
                        true = data.y.cpu().detach().numpy()
                        trues.append(true)

                        perm2 = outputs[-1].cpu().detach().numpy()
                        score2 = outputs[-3].cpu().detach().numpy()
                        perm2s.append(perm2)
                        score2s.append(score2) 
                    mean_community = np.array(weight_n2s).mean()
                    std_community = np.array(weight_n2s).std()
                    weights_n2s = np.array(weight_n2s)
                    # Find elements greater than both the mean and the std
                    fold_indices = []  
                    mask = (weights_n2s > mean_community) & (weights_n2s > std_community)
                    for i in range(mask.shape[0]):
                    # Get the indices of elements that satisfy the condition
                        indices = np.nonzero(mask[i])

        # Store indices as tuples of lists, suitable for multi-dimensional data
                        indexed_tuple = tuple(arr.tolist() for arr in indices)
                        fold_indices.append(indexed_tuple)
                    interesting_indices[fold] = fold_indices 


                    preds = np.concatenate(preds,axis=0)
                    trues = np.concatenate(trues,axis=0)
                    prabas = np.concatenate(prabas,axis=0)
                    cm = confusion_matrix(trues,preds)
                    tn, fp, fn, tp = cm.ravel()
                    results['Accuracy'].append(accuracy_score(trues, preds))
                    results['ROC AUC'].append(roc_auc_score(trues, prabas))
                    results['Specificity'].append(tn / (tn + fp))
                    results['Sensitivity'].append(tp / (tp + fn))
                    

                else:
                    if opt.nclass == 4:
                        model.load_state_dict(best_model_wts)
                        model.eval()
                        test_accuracy = test_acc(test_loader)
                        test_l= test_loss(test_loader,0)
                        print("===========================")
                        print("connectome {} classpair {} Test Acc: {:.7f}, Test Loss: {:.7f} ".format(connectome, class_pair, test_accuracy, test_l))
                        preds = []
                        prabas = []
                        trues = []
                        weight_n1s = []
                        weight_n2s = []
                        perm2s = []
                        score2s = []

                        for data in test_loader:
                            data = data.to(device)
                            outputs= model(data.x.float(), data.edge_index, data.batch, data.edge_attr.float(),data.pos)
                            pred = outputs[0].max(1)[1]
                            praba = outputs[0].exp()
                            prabas.append(praba.cpu().detach().numpy())
                            preds.append(pred.cpu().detach().numpy())
                            true = data.y.cpu().detach().numpy()
                            trues.append(true)
                            weight_n1 = model.n1[0].weight.data
                            weight_n2 = model.n2[0].weight.data

                            weight_n1s.append(weight_n1.cpu().detach().numpy())
                            weight_n2s.append(weight_n2.cpu().detach().numpy())
                            prabas.append(praba.cpu().detach().numpy())
                            preds.append(pred.cpu().detach().numpy())
                            true = data.y.cpu().detach().numpy()
                            trues.append(true)
                                # Capture perm2 and score2
                            perm2 = outputs[-1].cpu().detach().numpy()
                            score2 = outputs[-3].cpu().detach().numpy()
                            perm2s.append(perm2)
                            score2s.append(score2)                    
                        mean_community = np.array(weight_n2s).mean()
                        std_community = np.array(weight_n2s).std()
                        weights_n2s = np.array(weight_n2s)
                        # Find elements greater than both the mean and the std
                        fold_indices = []  
                        mask = (weights_n2s > mean_community) & (weights_n2s > std_community)
                        for i in range(mask.shape[0]):
                        # Get the indices of elements that satisfy the condition
                            indices = np.nonzero(mask[i])

            # Store indices as tuples of lists, suitable for multi-dimensional data
                            indexed_tuple = tuple(arr.tolist() for arr in indices)
                            fold_indices.append(indexed_tuple)
                        interesting_indices[fold] = fold_indices 


                        preds = np.concatenate(preds,axis=0)
                        trues = np.concatenate(trues,axis=0)
                        prabas = np.concatenate(prabas,axis=0)
                        n_classes = opt.nclass
                        cm = confusion_matrix(trues,preds,labels=[label for label in range(n_classes)])

                        trues_binarized = label_binarize(trues, classes=np.arange(n_classes))
                        # tn, fp, fn, tp = cm.ravel()
                        results['Accuracy'].append(accuracy_score(trues, preds))
                        # Alternatively, for One-vs-One approach
                        roc_auc_ovo = roc_auc_score(trues_binarized, prabas, multi_class='ovo')
                        results['ROC AUC'].append(roc_auc_ovo)


                        for i in range(1, n_classes):  # Start at 1 to compare class 0 with each other class
                            TP = cm[0, 0]
                            FN = cm[0, i]
                            FP = cm[i, 0]
                            TN = cm.sum() - (cm[0, :].sum() + cm[:, i].sum() - TP)

                            specificity_i = TN / (TN + FP) if (TN + FP) != 0 else 0
                            sensitivity_i = TP / (TP + FN) if (TP + FN) != 0 else 0

                            results['Specificity'].append(specificity_i)
                            results['Sensitivity'].append(sensitivity_i)

                            class_specific_results.append({
                                'Class_0_vs': i,
                                'Specificity': specificity_i,
                                'Sensitivity': sensitivity_i
                            })
                        # Display class-wise results
                        print("Class-specific metrics:", class_specific_results)


                    else:
                        preds = []
                        prabas = []
                        trues = []
                        weight_n1s = []
                        weight_n2s = []
                        perm1s = []
                        score1s = []
                        perm2s = []
                        score2s = []
                        correct = 0
                        for data in test_loader:
                            data = data.to(device)
                            outputs= model(data.x.float(), data.edge_index, data.batch, data.edge_attr.float(),data.pos)
                            pred = outputs[0].max(1)[1]
                            praba = outputs[0].exp()[:, 1] 
                            prabas.append(praba.cpu().detach().numpy())
                            preds.append(pred.cpu().detach().numpy())
                            correct += pred.eq(data.y).sum().item()
                            true = data.y.cpu().detach().numpy()
                            trues.append(true)
                            weight_n1 = model.n1[0].weight.data
                            weight_n2 = model.n2[0].weight.data

                            weight_n1s.append(weight_n1.cpu().detach().numpy())
                            weight_n2s.append(weight_n2.cpu().detach().numpy())

                                # Capture perm2 and score2
                            perm2 = outputs[-1].cpu().detach().numpy()
                            perm1 = outputs[-2].cpu().detach().numpy()

                            score2 = outputs[-3].cpu().detach().numpy()
                            score1 = outputs[-4].cpu().detach().numpy()

                            perm1s.append(perm1)
                            score1s.append(score1)

                            perm2s.append(perm2)
                            score2s.append(score2)

                        mean_community = np.array(weight_n2s).mean()
                        std_community = np.array(weight_n2s).std()
                        weights_n2s = np.array(weight_n2s)
                        # Find elements greater than both the mean and the std
                        fold_indices = []  
                        mask = (weights_n2s > mean_community) & (weights_n2s > std_community)
                        for i in range(mask.shape[0]):
                        # Get the indices of elements that satisfy the condition
                            indices = np.nonzero(mask[i])

            # Store indices as tuples of lists, suitable for multi-dimensional data
                            indexed_tuple = tuple(arr.tolist() for arr in indices)
                            fold_indices.append(indexed_tuple)
                        interesting_indices[fold] = fold_indices 

                        preds = np.concatenate(preds,axis=0)
                        trues = np.concatenate(trues,axis=0)
                        prabas = np.concatenate(prabas,axis=0)
                        cm = confusion_matrix(trues,preds)
                        tn, fp, fn, tp = cm.ravel()
                        results['Accuracy'].append(accuracy_score(trues, preds))
                        results['ROC AUC'].append(roc_auc_score(trues, prabas))
                        results['Specificity'].append(tn / (tn + fp))
                        results['Sensitivity'].append(tp / (tp + fn))
                        print(results)
            # Assuming perm2s is a list of arrays of different sizes
            df = pd.DataFrame({'ArrayData first pool': perm1s})

            # Save as a CSV or Excel file
            df.to_csv(f"{opt.save_path_results}/perm1.csv", index=False)
            # Assuming perm2s is a list of arrays of different sizes
            df = pd.DataFrame({'ArrayData first pool': score1s})

            # Save as a CSV or Excel file
            df.to_csv(f"{opt.save_path_results}/score2s.csv", index=False)
            avg_results = {matric: np.mean(values) for matric, values in results.items()} 
            avg_results = pd.DataFrame([avg_results])
            avg_results.to_csv(osp.join(opt.save_path_results,'Avarege results across folds {} threshold {}'.format(class_pair, percentile)))
            print(f"results saved in folder {current_time} {percentile} _ {class_pair}")
            with open('interesting_indices.json', 'w') as f:
                json.dump(interesting_indices, f)

            class_results_df = pd.DataFrame(class_specific_results)
            class_results_path = os.path.join(opt.save_path_results, f'Class_specific_results_{class_pair}_{percentile}.csv')
            class_results_df.to_csv(class_results_path, index=False)