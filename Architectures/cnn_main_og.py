import sys, os
import numpy as np
import json, argparse, sys
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.append('src') 
from load_data import Data
#from parameter import Parameter
# from linear_net import LinNet
from nn_gen import CNNNet

pwd = os.getcwd()


def plot_results(loss_arr, res_path, res_name):
    num_epochs= len(loss_arr)

    # Plot saved in results directory
    plt.plot(range(1, num_epochs+1), loss_arr)
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross Entropy Loss")
    plt.grid()
    plt.savefig(os.path.join(res_path, res_name))
    plt.close()


def prep(input_folder, params, view):
    # Construct a model
    cnn_net = CNNNet(view).to(torch.device("cpu"))

    # Construct data
    data = Data(input_folder, params['training']['batch size'], params['training']['batch size'], view)
    train_loader, test_loader = data.loaders()
    
    train_batchlists = list(train_loader)
    test_batchlists = list(test_loader)

    return cnn_net, train_batchlists, test_batchlists


def run(cnn, train_batchlists, test_batchlists, params):
    # Define an optimizer and the loss function
    optimizer = optim.Adam(cnn.parameters(), lr=params['optim']['learning rate'], betas=(params['optim']['beta 1'], params['optim']['beta 2']), eps=params['optim']['epsilon'], weight_decay=params['optim']['weight decay'])
    #optimizer = optim.Adam(cnn.parameters(), lr=1e-6)
    loss_fn = nn.BCELoss()

    num_epochs= int(params['training']['num epochs'])
    #num_epochs= 5

    train_loss_arr, test_loss_arr = [], []

    cnn.train()

    for epoch in range(num_epochs):

        inputs, labels = train_batchlists[epoch][0], train_batchlists[epoch][1]

        labels = torch.unsqueeze(labels, dim=1)

        outputs = cnn(inputs)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_arr.append(loss)
        
        if params['output']['verbosity']:
            print('Epoch [{}/{}]:'.format(epoch+1, num_epochs))
            print('Training Loss: {:.4f}'.format(loss.item()))
            print()

    return cnn, train_loss_arr, test_loss_arr



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN Architecture')

    parser.add_argument('--view', type=str, default = 'local',help="Input 'local','global', or 'both'")
    parser.add_argument('--param',type=str,default='param_local.json',help='location of params file')
    parser.add_argument('--input',type=str,default='torch_data',help='location of folder for data')
    args = parser.parse_args()

    param_file = args.param
    with open(os.path.join('param', param_file)) as paramfile:
        params = json.load(paramfile)
    #params = Parameter(param_file,pwd)
    input_folder = args.input
    view = args.view

    # Prepare CNN model and training/testing data
    cnn, train_batchlists, test_batchlists = prep(input_folder, params, view)

    # Train model and obtain train/test loss values
    trained_cnn, train_loss_arr, test_loss_arr = run(cnn, train_batchlists, test_batchlists, params)

    # Plot train/test loss values
    plot_results(train_loss_arr, params['output']['results path'], params['output']['results name'])
    #plot_results(train_loss_arr, 'results', 'fig_test_1.pdf')
    
    
    ## Example of how to use it with epochs
    #epoch_num = 2
    #for e in range(epoch_num):
    #    inputs, labels = train_batchlists[e][0], train_batchlists[e][1]
    #    print(inputs)
    #    print(labels)
        
#     lin_net = LinNet(view)
    
