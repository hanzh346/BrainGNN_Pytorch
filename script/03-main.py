import os
import numpy as np
import argparse
import time
import copy
from datetime import datetime
from Glioma_data_2 import *
import os 
import sys
parent_dir = '/home/hang/GitHub/BrainGNN_Pytorch'
sys.path.append(os.path.join(parent_dir, 'net'))
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from braingnn import Network
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from utils import *

def main():
    torch.manual_seed(123)
       # Define the base directory where your data is organized
    base_directory = '/media/hang/EXTERNAL_US/Data/glioma/data/organized data/pre_surgery'
    EPS = 1e-10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    parser = argparse.ArgumentParser()
    # ... [Include all your existing argument parsers here] ...

    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=16, help='size of the batches')
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
    parser.add_argument('--indim', type=int, default=1, help='node feature dim')
    parser.add_argument('--nroi', type=int, default=116, help='num of ROIs')
    parser.add_argument('--nclass', type=int, default=2, help='num of classes')
    parser.add_argument('--balance', type=bool, default=True, help='Whether to balance the dataset by downsampling')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--top_percent', type=float, default=0.1, help='Top percentage for pruning adjacency matrix')
    parser.add_argument('--save_path', type=str, default='./model/', help='path to save model')
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--n_splits', type=int, default=5, help='number of cross-validation folds')  # Added argument
    parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
    opt = parser.parse_args()
    
    # Get current date and time for saving results
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Update save paths to include the date
    model_save_path = os.path.join(opt.save_path, current_datetime)
    log_save_path = os.path.join('./log', current_datetime)
    
    # Create directories if they don't exist
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)
    
    #################### Parameter Initialization #######################
    name = 'Glioma'
    save_model = opt.save_model
    load_model = opt.load_model
    opt_method = opt.optim
    num_epoch = opt.n_epochs
    n_splits = opt.n_splits
    balance = opt.balance
    random_state = opt.random_state
    top_percent = opt.top_percent
    
    # Initialize the main SummaryWriter for overall run (optional)
    main_writer = SummaryWriter(log_save_path)
    
    ################## Define Dataloader ##################################
    
 
    
    # Step 1: Load Data with Balancing
    data_list = load_data_with_graph_attributes(
        base_dir=base_directory,
        top_percent=top_percent,
        balance=balance,
        random_state=random_state,
        density = top_percent
    )
    
    print(f"Total samples after balancing: {len(data_list)}")
    
    # Extract labels for StratifiedKFold
    labels = np.array([data.y.item() for data in data_list])
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []  # To store metrics per fold
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\nStarting Fold {fold + 1}/{n_splits}")
    
        # Split data
        train_dataset = [data_list[i] for i in train_idx]
        val_dataset = [data_list[i] for i in val_idx]
    
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
    
        batch_size = opt.batchSize  # Use argparse batch size
    
        # Create DataLoader instances
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    

        ############### Define Graph Deep Learning Network ##########################
        model = Network(opt.indim, opt.ratio, opt.nclass, opt.nroi).to(device)
        print(model)
    
        if opt_method == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weightdecay)
        elif opt_method == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weightdecay, nesterov=True)
    
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)
    
        writer = SummaryWriter(os.path.join(log_save_path, f'fold_{fold+1}'))  # Initialize writer per fold
    

    
        ###################### Network Testing Function#####################################
        def test_metrics(loader):
            """
            Evaluate the model on the provided data loader and compute various performance metrics,
            including accuracy, AUC-ROC, sensitivity, specificity, and the confusion matrix.
            Additionally, determine and apply an optimal threshold to balance sensitivity and specificity.
            
            Args:
                loader (DataLoader): PyTorch Geometric DataLoader containing the evaluation dataset.
            
            Returns:
                accuracy (float): Overall accuracy of the model.
                auc (float): Area Under the Receiver Operating Characteristic Curve.
                sensitivity (float): True Positive Rate.
                specificity (float): True Negative Rate.
                cm (numpy.ndarray): Confusion matrix.
                all_labels (list): Ground truth labels.
                preds_adjusted (numpy.ndarray): Predictions after applying the optimal threshold.
                all_probs (list): Predicted probabilities for the positive class.
                best_threshold (float): The optimal threshold determined using Youden's J statistic.
            """
            model.eval()
            correct = 0
            all_labels = []
            all_preds = []
            all_probs = []  # For AUC-ROC
    
            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    output, w1, w2, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
                    probs = torch.exp(output)  # Convert log-probabilities to probabilities
                    preds = output.argmax(dim=1)
                    correct += preds.eq(data.y).sum().item()
    
                    all_labels.extend(data.y.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
            accuracy = correct / len(loader.dataset)
            
            # Calculate AUC-ROC
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                auc = float('nan')  # Handle cases where AUC is not defined
    
            # Determine the optimal threshold using Youden's J statistic
            try:
                fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
                youden_index = tpr - fpr
                best_idx = np.argmax(youden_index)
                best_threshold = thresholds[best_idx]
            except ValueError:
                best_threshold = 0.5  # Default threshold if unable to compute
    
            # Apply the optimal threshold to obtain adjusted predictions
            preds_adjusted = (np.array(all_probs) >= best_threshold).astype(int)
    
            # Calculate confusion matrix based on adjusted predictions
            cm = confusion_matrix(all_labels, preds_adjusted)
    
            # Extract True Negatives (TN), False Positives (FP), False Negatives (FN), and True Positives (TP)
            if cm.shape == (2, 2):
                TN, FP, FN, TP = cm.ravel()
                sensitivity = TP / (TP + FN) if (TP + FN) > 0 else float('nan')
                specificity = TN / (TN + FP) if (TN + FP) > 0 else float('nan')
            else:
                # Handle cases where there are not exactly two classes
                sensitivity = float('nan')
                specificity = float('nan')
    
            return accuracy, auc, sensitivity, specificity, cm, all_labels, preds_adjusted, all_probs, best_threshold
    
        def test_loss(loader, model, opt):
            print('testing...........')
            model.eval()
            loss_all = 0
            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    output, w1, w2, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
                    loss_c = F.nll_loss(output, data.y)
    
                    loss_p1 = (torch.norm(w1, p=2) - 1) ** 2
                    loss_p2 = (torch.norm(w2, p=2) - 1) ** 2
                    loss_tpk1 = topk_loss(s1, opt.ratio)
                    loss_tpk2 = topk_loss(s2, opt.ratio)
                    loss_consist = 0
                    for c in range(opt.nclass):
                        loss_consist += consist_loss(s1[data.y == c])
                    loss = (opt.lamb0 * loss_c +
                            opt.lamb1 * loss_p1 +
                            opt.lamb2 * loss_p2 +
                            opt.lamb3 * loss_tpk1 +
                            opt.lamb4 * loss_tpk2 +
                            opt.lamb5 * loss_consist)
    
                    loss_all += loss.item() * data.num_graphs
            return loss_all / len(loader.dataset)
    
        #######################################################################################
        ############################   Model Training #########################################
        #######################################################################################
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 1e10
        for epoch in range(0, num_epoch):
            since = time.time()
            tr_loss, s1_arr, s2_arr, w1, w2 = train(epoch,scheduler,optimizer,model,train_loader,opt,writer,train_dataset)
            
            # Test on training data
            tr_acc, tr_auc, tr_sens, tr_spec, tr_cm, tr_labels, tr_preds_adjusted, tr_probs, tr_best_threshold = test_metrics(train_loader)
            
            # Test on validation data
            val_acc, val_auc, val_sens, val_spec, val_cm, val_labels, val_preds_adjusted, val_probs, val_best_threshold = test_metrics(val_loader)
            
            # Compute validation loss
            val_loss = test_loss(val_loader, model, opt)  # Existing loss calculation
            
            time_elapsed = time.time() - since
            print('*====**')
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Epoch: {:03d}, Train Loss: {:.7f}, '
                  'Train Acc: {:.7f}, Val Loss: {:.7f}, Val Acc: {:.7f}'.format(
                      epoch, tr_loss, tr_acc, val_loss, val_acc))
    
            writer.add_scalars('Acc', {'train_acc': tr_acc, 'val_acc': val_acc}, epoch)
            writer.add_scalars('Loss', {'train_loss': tr_loss, 'val_loss': val_loss}, epoch)
            writer.add_histogram('Hist/hist_s1', s1_arr, epoch)
            writer.add_histogram('Hist/hist_s2', s2_arr, epoch)
    
            # Log additional metrics
            writer.add_scalar('train/AUC_ROC', tr_auc, epoch)
            writer.add_scalar('train/Sensitivity', tr_sens, epoch)
            writer.add_scalar('train/Specificity', tr_spec, epoch)
            writer.add_scalar('val/AUC_ROC', val_auc, epoch)
            writer.add_scalar('val/Sensitivity', val_sens, epoch)
            writer.add_scalar('val/Specificity', val_spec, epoch)
            writer.add_scalar('val/Optimal_Threshold', val_best_threshold, epoch)  # Log the optimal threshold
    
            # Optionally, visualize the ROC curve for validation set
            fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
    
            if val_loss < best_loss and epoch > 5:
                print("Saving best model")
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                if save_model:
                    model_filename = f'fold_{fold+1}.pth'
                    torch.save(best_model_wts, os.path.join(model_save_path, model_filename))
    
        writer.close()
    
        #######################################################################################
        ######################### Testing on validation set ######################################
        #######################################################################################
    
        if opt.load_model:
            # Load the saved model
            model_path = os.path.join(model_save_path, f"fold_{fold+1}.pth")
            if not os.path.exists(model_path):
                print(f"Model file not found at {model_path}.")
            else:
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                print(f"Loaded model from {model_path} and set to evaluation mode.")
    
                preds = []
                trues = []
                probs = []  # For AUC-ROC
                with torch.no_grad():
                    for data in val_loader:
                        data = data.to(device)
                        output, w1, w2, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
                        prob = torch.exp(output)[:, 1]  # Probability of positive class
                        pred = output.argmax(dim=1)
                        preds.append(pred.cpu().numpy())
                        trues.append(data.y.cpu().numpy())
                        probs.append(prob.cpu().numpy())  # Collect probabilities
    
                preds = np.concatenate(preds, axis=0)
                trues = np.concatenate(trues, axis=0)
                probs = np.concatenate(probs, axis=0)
    
                # Determine the optimal threshold using Youden's J statistic
                try:
                    fpr, tpr, thresholds = roc_curve(trues, probs)
                    youden_index = tpr - fpr
                    best_idx = np.argmax(youden_index)
                    best_threshold = thresholds[best_idx]
                except ValueError:
                    best_threshold = 0.5  # Default threshold if unable to compute
    
                # Apply the optimal threshold to obtain adjusted predictions
                preds_adjusted = (probs >= best_threshold).astype(int)
    
                # Calculate metrics based on adjusted predictions
                cm = confusion_matrix(trues, preds_adjusted)
                if cm.shape == (2, 2):
                    TN, FP, FN, TP = cm.ravel()
                    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else float('nan')
                    specificity = TN / (TN + FP) if (TN + FP) > 0 else float('nan')
                else:
                    sensitivity = float('nan')
                    specificity = float('nan')
    
                try:
                    auc = roc_auc_score(trues, probs)
                except ValueError:
                    auc = float('nan')
    
                print("Confusion Matrix:")
                print(cm)
                print("\nClassification Report:")
                print(classification_report(trues, preds_adjusted))
                print(f"AUC-ROC: {auc:.4f}")
                print(f"Sensitivity: {sensitivity:.4f}")
                print(f"Specificity: {specificity:.4f}")
                print(f"Optimal Threshold: {best_threshold:.4f}")  # Print the optimal threshold
    
                # Optionally, store fold results
                fold_results.append({
                    'fold': fold + 1,
                    'confusion_matrix': cm,
                    'classification_report': classification_report(trues, preds_adjusted, output_dict=True),
                    'AUC_ROC': auc,
                    'Sensitivity': sensitivity,
                    'Specificity': specificity,
                    'best_threshold': best_threshold  # Store the optimal threshold
                })
    
        else:
            # Evaluate on validation set with best model
            model.load_state_dict(best_model_wts)
            model.eval()
            test_acc_val, test_auc_val, test_sens_val, test_spec_val, test_cm_val, test_labels_val, test_preds_adjusted_val, test_probs_val, test_best_threshold_val = test_metrics(val_loader)
            test_l = test_loss(val_loader, model, opt)
            print("===========================")
            print("Fold {} - Test Acc: {:.7f}, Test Loss: {:.7f}, AUC-ROC: {:.4f}, Sensitivity: {:.4f}, Specificity: {:.4f}".format(
                fold + 1, test_acc_val, test_l, test_auc_val, test_sens_val, test_spec_val))  # Modified
            print(f"Optimal Threshold: {test_best_threshold_val:.4f}")  # Print the optimal threshold
            print(opt)
    
            # Store fold results
            fold_results.append({
                'fold': fold + 1,
                'test_accuracy': test_acc_val,
                'test_loss': test_l,
                'AUC_ROC': test_auc_val,
                'Sensitivity': test_sens_val,
                'Specificity': test_spec_val,
                'best_threshold': test_best_threshold_val  # Store the optimal threshold
            })
    
        # After all folds
        if not opt.load_model:
            # Calculate average metrics
            avg_acc = np.mean([result['test_accuracy'] for result in fold_results if 'test_accuracy' in result])
            avg_loss = np.mean([result['test_loss'] for result in fold_results if 'test_loss' in result])
            avg_auc = np.mean([result['AUC_ROC'] for result in fold_results if 'AUC_ROC' in result])
            avg_sens = np.mean([result['Sensitivity'] for result in fold_results if 'Sensitivity' in result])
            avg_spec = np.mean([result['Specificity'] for result in fold_results if 'Specificity' in result])
            avg_threshold = np.mean([result['best_threshold'] for result in fold_results if 'best_threshold' in result])
    
            print("\nCross-Validation Results:")
            print(f"Average Test Accuracy: {avg_acc:.7f}")
            print(f"Average Test Loss: {avg_loss:.7f}")
            print(f"Average AUC-ROC: {avg_auc:.4f}")
            print(f"Average Sensitivity: {avg_sens:.4f}")
            print(f"Average Specificity: {avg_spec:.4f}")
            print(f"Average Optimal Threshold: {avg_threshold:.4f}")  # Print the average optimal threshold
            print(opt)
    
            # Save summary to a text file
            summary_file = os.path.join(model_save_path, 'cross_validation_summary.txt')
            with open(summary_file, 'w') as f:
                f.write("Cross-Validation Results\n")
                f.write(f"Date and Time: {current_datetime}\n")
                f.write(f"Number of Splits: {n_splits}\n")
                f.write(f"Average Test Accuracy: {avg_acc:.7f}\n")
                f.write(f"Average Test Loss: {avg_loss:.7f}\n")
                f.write(f"Average AUC-ROC: {avg_auc:.4f}\n")
                f.write(f"Average Sensitivity: {avg_sens:.4f}\n")
                f.write(f"Average Specificity: {avg_spec:.4f}\n")
                f.write(f"Average Optimal Threshold: {avg_threshold:.4f}\n\n")  # Added
                for result in fold_results:
                    f.write(f"Fold {result['fold']}:\n")
                    if 'test_accuracy' in result:
                        f.write(f"  Test Accuracy: {result['test_accuracy']:.7f}\n")
                        f.write(f"  Test Loss: {result['test_loss']:.7f}\n")
                    if 'AUC_ROC' in result:
                        f.write(f"  AUC-ROC: {result['AUC_ROC']:.4f}\n")
                    if 'Sensitivity' in result:
                        f.write(f"  Sensitivity: {result['Sensitivity']:.4f}\n")
                    if 'Specificity' in result:
                        f.write(f"  Specificity: {result['Specificity']:.4f}\n")
                    if 'best_threshold' in result:
                        f.write(f"  Optimal Threshold: {result['best_threshold']:.4f}\n")
                    if 'confusion_matrix' in result:
                        f.write("  Confusion Matrix:\n")
                        f.write(np.array2string(result['confusion_matrix']))
                        f.write("\n")
                    if 'classification_report' in result:
                        f.write("  Classification Report:\n")
                        f.write(str(result['classification_report']))
                        f.write("\n\n")
            print(f"Cross-validation summary saved to {summary_file}")
    
        else:
            # If load_model was True, you might want to save classification reports or confusion matrices
            pass  # Implement aggregation if required
    
        main_writer.close()  # Close the main writer if used

if __name__ == "__main__":
    main()

