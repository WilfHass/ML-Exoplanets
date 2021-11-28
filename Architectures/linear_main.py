import sys, os
import numpy as np
import sys
import torch
import argparse
sys.path.append('src') 
from load_data import *
from parameter import Parameter
from linear_net import LinNet
from helper_gen import make_parser, performance, optimize
from heatmap import *

pwd = os.getcwd()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Main')
    parser.add_argument('--input', default ='src/torch_data',type=str, help="location for input file")
    parser.add_argument('--view', default='global',type=str,help="view")
    parser.add_argument('--param', default='param/linear_global.json',type=str,help="location of parameter file")
    parser.add_argument('--result', default='result/',type=str,help="location of parameter file")

   
    args = parser.parse_args()

    param_file = args.param

    params = Parameter(param_file, pwd)
    input_folder = args.input
    print(input_folder)
    view = args.view
    result_file = args.result
    epoch_num = params.epoch

    train_loader, test_loader = dataPrep(input_folder, params.trainbs,params.testbs)
    lin_net = LinNet(view)
    optim = torch.optim.Adam(lin_net.parameters(), lr=params.lr, betas=(0.9, 0.99), amsgrad=False)

    # optim = torch.optim.SGD(model.parameters(),lr=params.lr, momentum=params.mom)

    loss_fn = torch.nn.BCELoss()

    for e in range(epoch_num):
        
        # Train the model
        lin_net.train()
        for batch_idx, (data, label) in enumerate(train_loader):

            optim.zero_grad()
            outputs = lin_net(data)
            label = torch.reshape(label,(len(label),-1))
            loss = loss_fn(outputs, label)
            loss.backward()
            optim.step()
            

        if e % 10 == 0:
            print("Epoch [{}/{}] \t Loss: {}".format(e, epoch_num, loss.item()))

    
    #optimize(fc_net, train_batchlists, params)
    test_set = list(test_loader)
    create_heatmap(lin_net,input_folder)
    perf_list = performance(lin_net, test_set)