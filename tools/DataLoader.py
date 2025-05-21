# -*- coding: utf-8 -*-
import sys
homepath = "../../"
sys.path.append(homepath)
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def Standard(s, mean=None, sd=None):
    if mean is None:
        mean = np.mean(s, axis=0)
    if sd is None:
        sd = np.std(s, axis=0)
    s = (s - mean) / sd
    return s

def InvStandard(s, mean, sd):
    s = s * sd + mean
    return s

def getLoader(filename="data_w.csv", BATCH_SIZE=100):

    data = np.array(pd.read_csv(homepath + "data/dataset/" + filename).iloc[:, 1:])

    X = data[:, 3:6]
    s = data[:, 6:]

    mean_s, sd_s = np.mean(s), np.std(s)
    
    X = X / np.sum(X, axis=1, keepdims=True)
    X = X[:, :2]
    
    X = Standard(X)
    s = Standard(s)
    
    X_train, X_test, s_train, s_test = train_test_split(X, s, test_size=0.2,
                                                        random_state=10)
    X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
    s_train, s_test = torch.Tensor(s_train), torch.Tensor(s_test)
    
    train_dataset = torch.utils.data.TensorDataset(X_train, s_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, s_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)
    return train_loader, test_loader, mean_s, sd_s
