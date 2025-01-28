import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import pickle
from Glioma_data_2 import load_data_with_graph_attributes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define the custom Dataset
class MLPDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        Args:
            data_list (list): List of torch_geometric.data.Data objects.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (feature_tensor, label_tensor)
                - feature_tensor: torch.FloatTensor of shape [13456]
                - label_tensor: torch.LongTensor containing the label (0 or 1)
        """
        data = self.data_list[idx]
        feature = data.original_edge_attr.numpy()  # Convert to NumPy array
        label = data.y.item()  # Assuming label is a tensor with a single value

        if self.transform:
            feature = self.transform(feature)

        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return feature_tensor, label_tensor

# Define the MLP Model
class MLP(nn.Module):
    def __init__(self, input_dim=13456, hidden_dim=64, output_dim=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.8)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.fc4 = nn.Linear(hidden_dim//4, output_dim)
        
    def forward(self, x):
        out = self.fc1(x)       # [batch_size, hidden_dim]
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out) 
        out = self.relu(out)    # [batch_size, output_dim]
        out = self.fc3(out) 
        out = self.relu(out)
        out = self.fc4(out) 
        return out

# Function to initialize weights (optional)
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# Load data using your existing function
base_directory = '/media/hang/EXTERNAL_US/Data/glioma/data/organized data/pre_surgery'
density = 0.15  # Example density
balanced_data_list = load_data_with_graph_attributes(
    base_dir=base_directory,
    density=density,
    top_percent=0.15,
    balance=True,
    random_state=42
)


# Extract labels for StratifiedKFold
labels = np.array([data.y.item() for data in balanced_data_list])

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print(f"\nStarting Fold {fold + 1}/5")
    
    # Create training and validation subsets
    train_subset = [balanced_data_list[i] for i in train_idx]
    val_subset = [balanced_data_list[i] for i in val_idx]
    
    # Initialize the custom Dataset
    train_dataset = MLPDataset(train_subset)
    val_dataset = MLPDataset(val_subset)
    
    # Create DataLoaders using torch.utils.data.DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,        # Adjust based on your system
        pin_memory=True       # If using CUDA
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,        # Adjust based on your system
        pin_memory=True       # If using CUDA
    )
    
    # Initialize the MLP model, optimizer, and loss function
    model = MLP(input_dim=13456, hidden_dim=2048, output_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    # Training and Validation Loop
    num_epochs = 75
    patience = 15
    best_val_loss = float('inf')
    trigger_times = 0
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        for batch_idx, (features, labels_batch) in enumerate(train_loader):
            features = features.to(device)    # [batch_size, 13456]
            labels_batch = labels_batch.to(device)  # [batch_size]
            
            optimizer.zero_grad()
            outputs = model(features)         # [batch_size, 2]
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for features, labels_batch in val_loader:
                features = features.to(device)    # [batch_size, 13456]
                labels_batch = labels_batch.to(device)  # [batch_size]
                
                outputs = model(features)         # [batch_size, 2]
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)  # [batch_size]
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate Metrics
        if len(all_labels) > 0 and len(all_preds) > 0:
            cm = confusion_matrix(all_labels, all_preds)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                sensitivity, specificity = 0, 0  # Handle cases with missing classes
            accuracy = accuracy_score(all_labels, all_preds)
            try:
                roc_auc = roc_auc_score(all_labels, all_preds)
            except ValueError:
                roc_auc = 0  # Handle cases where ROC-AUC cannot be computed
        else:
            sensitivity, specificity, accuracy, roc_auc = 0, 0, 0, 0  # Default values
        
        # Print Metrics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Sensitivity: {sensitivity:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        
        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            # Save the best model for this fold
            torch.save(model.state_dict(), f'best_model_fold_{fold+1}.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Store Fold Results
    fold_results.append({
        "fold": fold + 1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "roc_auc": roc_auc
    })

# Summarize Results Across Folds
print("\nCross-Validation Results:")
for result in fold_results:
    print(f"Fold {result['fold']}: Sensitivity: {result['sensitivity']:.4f}, "
          f"Specificity: {result['specificity']:.4f}, "
          f"Accuracy: {result['accuracy']:.4f}, "
          f"ROC-AUC: {result['roc_auc']:.4f}")
