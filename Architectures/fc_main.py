import sys, os
import numpy as np
import json, argparse, sys
import csv
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
sys.path.append('src') 
from load_data import Data
from parameter import Parameter
from fc_net import FCNet
pwd = os.getcwd()

def performance(fin_model,test_set):
    inputs, labels = test_set[0][0],test_set[0][1]
    outputs = fin_model.forward(inputs)
    classification_threshold = 0.5
    for i in range(len(outputs)):
        if outputs[i]>=classification_threshold:
            outputs[i] = 1.0
        else:
            outputs[i] = 0.0 
    accuracy = 0
    for i in range(len(outputs)):
        if outputs[i] == labels[i]:
            accuracy+=1

    accuracy = accuracy/len(outputs)
    print("Final Accuracy: ",accuracy)
    
    label_arr = labels.detach().numpy()
    output_arr = outputs.detach().numpy()
    AUC = roc_auc_score(label_arr,output_arr)
    print(AUC)
    
    ## Precision
    count = 0
    for i in range(len(outputs)):
        if outputs[i]==1.0 and labels[i]==1.0:
            count+=1
        if labels[i]==1.0:
            total+=1
    precision = count/total 
    
    ## Recall 
    count = 0
    for i in range(len(outputs)):
        if outputs[i]==1.0 and labels[i]==1.0:
            count+=1
        if outputs[i]==1.0:
            total+=1
    recall = count/total 

def optimize(model,batchlist,params):
    epoch_num = params.epoch
    optim = torch.optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.99), amsgrad=False)
#     optim = torch.optim.SGD(model.parameters(),lr=params.lr, momentum=params.mom)
    loss_fn = torch.nn.BCELoss()
    for e in range(epoch_num):
        for data in batchlist:
            inputs, labels = data[0], data[1]
            outputs = model.forward(inputs)
            r_outputs = torch.reshape(outputs,(-1,))
            loss = loss_fn(r_outputs,labels)
            loss.backward()
            optim.step()
            optim.zero_grad()
        if e%100 == 0:
            print(e,loss.item())
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FC Architecture')

    parser.add_argument('--view', type=str, default = 'local',help="Input 'local','global', or 'both'")
    parser.add_argument('--param',type=str,default='param/fc_params_local.json',help='location of params file')
    parser.add_argument('--input',type=str,default='torch_data',help='location of folder for data')
    args = parser.parse_args()

    param_file = args.param
    params = Parameter(param_file,pwd)
    input_folder = args.input
    view = args.view
    
    data = Data(input_folder,params.trainbs,params.testbs,view)
    train_loader, test_loader = data.loaders()
    test_set = list(test_loader)
    
    train_batchlists = list(train_loader)
    
    size = len(train_batchlists[0][0][0])
    fc_net = FCNet(size,view)
    
    optimize(fc_net,train_batchlists,params)
    performance(fc_net,test_set)
    
    
    