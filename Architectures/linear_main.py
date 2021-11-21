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
from linear_net import LinNet
pwd = os.getcwd()

def precision(outputs,labels):
    count = 0
    total = 0
    for i in range(len(outputs)):
        if outputs[i]==1.0 and labels[i]==1.0:
            count+=1
        if labels[i]==1.0:
            total+=1
    return(count/total)
def recall(outputs, labels):
    count = 0
    total = 0
    for i in range(len(outputs)):
        if outputs[i]==1.0 and labels[i]==1.0:
            count+=1
        if outputs[i]==1.0:
            total+=1
    return (count/total)      
def performance(fin_model,test_set):
    inputs, labels = test_set[0][0],test_set[0][1]
    outputs = fin_model.forward(inputs)
    
    ct = np.arange(0.3,0.9,0.05)
    pre_list = []
    rec_list = []
    
    for c in ct:
        new_out = np.zeros(len(outputs))
        for i in range(len(outputs)):
            if outputs[i]>=c:
                new_out[i] = 1.0
            else:
                new_out[i] = 0.0 
        prec = precision(new_out,labels) 
        rec = recall(new_out,labels)
        pre_list.append(prec)
        rec_list.append(rec)
    print(pre_list,rec_list)
    plt.plot(rec_list,pre_list)
    plt.show()
    plt.savefig("pre-rec.png")                    
                        
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
    
    ## AUC
    label_arr = labels.detach().numpy()
    output_arr = outputs.detach().numpy()
    AUC = roc_auc_score(label_arr,output_arr)
    print(AUC)
    


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
    parser = argparse.ArgumentParser(description='Linear Architecture')

    parser.add_argument('--view', type=str, default = 'local',help="Input 'local','global', or 'both'")
    parser.add_argument('--param',type=str,default='param/linear_params_local.json',help='location of params file')
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
    lin_net = LinNet(size)
    optimize(lin_net,train_batchlists,params)
    performance(lin_net,test_set)
    
    
    