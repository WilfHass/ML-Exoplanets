import sys, os
import sys
import torch
import argparse
import numpy as np
sys.path.append('src') 
from load_data import *
from parameter import Parameter
from fc_net import FCNet
from helper_gen import make_parser, performance, optimize
from heatmap import *

pwd = os.getcwd()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FC Main')
    parser.add_argument('--input', default ='src/torch_data',type=str, help="location for input file")
    parser.add_argument('--view', default='local',type=str,help="view")
    parser.add_argument('--param', default='param/fc_local.json',type=str,help="location of parameter file")
    parser.add_argument('--result', default='result/',type=str,help="location of parameter file")

   
    args = parser.parse_args()

    param_file = args.param

    params = Parameter(param_file, pwd)
    input_folder = args.input
    view = args.view
    result_file = args.result
    epoch_num = params.epoch

    train_loader, test_loader = dataPrep(input_folder, params.trainbs,params.testbs)
    test_set = list(test_loader)
    fc_net = FCNet(view)
    optim = torch.optim.Adam(fc_net.parameters(), lr=params.lr, betas=(0.9, 0.99), amsgrad=False)

    # optim = torch.optim.SGD(model.parameters(),lr=params.lr, momentum=params.mom)

    loss_fn = torch.nn.BCELoss()

    for e in range(epoch_num):
        
        # Train the model
        fc_net.train()
        loss_avg = [] 
        for batch_idx, (data, label) in enumerate(train_loader):

            optim.zero_grad()
            outputs = fc_net(data)
            label = torch.reshape(label,(len(label),-1))
            loss = loss_fn(outputs, label)
            loss.backward()
            optim.step()
            loss_avg.append(loss.item())
            

        if e % 10 == 0:
            print("Epoch [{}/{}] \t Loss: {}".format(e, epoch_num, np.mean(loss_avg)))
            performance(fc_net, test_set)

    
    #optimize(fc_net, train_batchlists, params)

    create_heatmap(fc_net, input_folder,view)
    perf_list = performance(fc_net, test_set)