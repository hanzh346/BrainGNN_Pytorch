import os
import numpy as np
import argparse
import time
import copy
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from braingnn import Network
from sklearn.metrics import classification_report, confusion_matrix

EPS = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
    ############################### Define Other Loss Functions ########################################

def topk_loss(s, ratio):
    if ratio > 0.5:
        ratio = 1 - ratio
    s = s.sort(dim=1).values
    res = -torch.log(s[:, -int(s.size(1)*ratio):] + EPS).mean() - torch.log(1 - s[:, :int(s.size(1)*ratio)] + EPS).mean()
    return res

def consist_loss(s):
    if len(s) == 0:
        return 0
    s = torch.sigmoid(s)
    W = torch.ones(s.shape[0], s.shape[0])
    D = torch.eye(s.shape[0]) * torch.sum(W, dim=1)
    L = D - W
    L = L.to(device)
    res = torch.trace(torch.transpose(s, 0, 1) @ L @ s) / (s.shape[0] * s.shape[0])
    return res
    
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

def test_loss(loader,model,opt):
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
###################### Network Training Function#####################################
def train(epoch,scheduler,optimizer,model,train_loader,opt,writer,train_dataset):
    print('train...........')
    scheduler.step()

    for param_group in optimizer.param_groups:
        print("LR", param_group['lr'])
    model.train()
    s1_list = []
    s2_list = []
    loss_all = 0
    step = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, w1, w2, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos)
        s1_list.append(s1.view(-1).detach().cpu().numpy())
        s2_list.append(s2.view(-1).detach().cpu().numpy())

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
        writer.add_scalar('train/classification_loss', loss_c, epoch * len(train_loader) + step)
        writer.add_scalar('train/unit_loss1', loss_p1, epoch * len(train_loader) + step)
        writer.add_scalar('train/unit_loss2', loss_p2, epoch * len(train_loader) + step)
        writer.add_scalar('train/TopK_loss1', loss_tpk1, epoch * len(train_loader) + step)
        writer.add_scalar('train/TopK_loss2', loss_tpk2, epoch * len(train_loader) + step)
        writer.add_scalar('train/GCL_loss', loss_consist, epoch * len(train_loader) + step)
        step += 1

        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    s1_arr = np.hstack(s1_list)
    s2_arr = np.hstack(s2_list)
    return loss_all / len(train_dataset), s1_arr, s2_arr, w1, w2