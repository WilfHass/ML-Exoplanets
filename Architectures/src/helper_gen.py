import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
import torch

def make_parser():
    parser = argparse.ArgumentParser(description='Exoplanet detection CNN: Local or global data only')
    parser.add_argument('--view', type=str, default = 'local', help="Input: 'local' | 'global' | 'both'")
    parser.add_argument('--param', type=str, default="param/fc_params_local.json", help="file name for json attributes.")
    parser.add_argument('--input', type=str, default='../data/torch_data', help='location of folder for data')
    parser.add_argument('--result', type=str, default='results/output1.txt',help='filename to save the test outputs at')

    args = parser.parse_args()

    if not os.path.exists('plots'):
        os.makedirs('plots')

    if not os.path.exists('results'):
        os.makedirs('results')

    return args


def precision(outputs,labels): 
    '''
    Computes the precision for a given output and label set
    '''
    count = 0
    for i in range(len(outputs)):
        if outputs[i]==1.0 and labels[i]==1.0:
            count+=1
#     count = sum(np.equal(outputs, labels))
    total = sum(labels)
    if total == 0:
        prec = 0
    else:
        prec = count/total
    return (prec)


def recall(outputs, labels):
    '''
    Computes the recall for a given output and label set. 
    outputs : classified TCEs e.g. 1 or 0
    labels : Actual values for TCEs e.g. 1 or 0
    '''
    count = 0
    for i in range(len(outputs)):
        if outputs[i]==1.0 and labels[i]==1.0:
            count+=1

    total = sum(outputs)
    if total == 0:
        rec = 0
    else:
        rec = count/total
    return (rec)


def accuracy(outputs, labels):
    '''
    Computes the accuracy for a given output and label set
    '''
    count = sum(np.equal(outputs, labels))
    accuracy = count/len(outputs)

    return accuracy


def is_TCE(outputs, classification_threshold):
    '''
    Turns each output into a definite classification
    outputs : finished model using the test data set
    classification_threshold : specific threshold to classify a TCE 
    '''
    new_out = outputs >= classification_threshold # boolean mask

    return new_out


def compare_thresholds(outputs, labels, out_file):
    '''
    Plots the precision vs recall for different classification thresholds.
    outputs : finished model using test set data
    labels : test set actual values
    '''
    # List of classification thresholds to run through
    ct = np.arange(0.3, 0.8, 0.01)

    # Precision and recall lists
    pre_list = []
    rec_list = []

    for c in ct:
        new_out = is_TCE(outputs, c)

        prec = precision(new_out, labels) 
        rec = recall(new_out, labels)
        if rec!=0 and prec!=0:
            pre_list.append(prec)
            rec_list.append(rec)

#     print(pre_list, rec_list)

    plt.plot(rec_list, pre_list)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.savefig(os.path.join('plots', out_file))
    plt.show()
    plt.close()


def performance(fin_model, test_set):
    '''
    Performance of the finished FCNN model using the test dataset. Calculates the Accuracy, Precision, AUC, Recall
    fin_model : Last model of the FCNN
    test_set : set of data to test the finished model
    '''
    

    inputs, labels = test_set[0][0], test_set[0][1]
    outputs = fin_model.forward(inputs)

    classification_threshold = 0.5

    labels = labels.detach().numpy().flatten()
    outputs = outputs.detach().numpy().flatten()

    # Classify the outputs
    classified_outputs = is_TCE(outputs, classification_threshold)
    # compare_thresholds(outputs, labels)
    
    # Accuracy
    acc = accuracy(classified_outputs, labels)
    # print("Final Accuracy: ", acc)
    
    # AUC
    AUC = roc_auc_score(labels, classified_outputs)
    # print("AUC: ", AUC)
    
    # Precision
    prec = precision(classified_outputs, labels)
    # print("Final Precision: ", prec)
    
    # Recall 
    rec = recall(classified_outputs, labels)
    # print("Final Recall: ", rec)

    return [acc, prec, rec, AUC]


def optimize(model, batchlist, params):

    epoch_num = params.epoch
    optim = torch.optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.99), amsgrad=False)
    # optim = torch.optim.SGD(model.parameters(),lr=params.lr, momentum=params.mom)
    loss_fn = torch.nn.BCELoss()

    for e in range(epoch_num):
        
        for data in batchlist:
            inputs, labels = data[0], data[1]
            outputs = model.forward(inputs)
            r_outputs = torch.reshape(outputs, (-1,))
            loss = loss_fn(r_outputs,labels)
            loss.backward()
            optim.step()
            optim.zero_grad()

        if e % 10 == 0:
            print("Epoch [{}/{}] \t Loss: {}".format(e, epoch_num, loss.item()))