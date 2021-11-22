import sys, os
import sys
import torch

sys.path.append('src') 
from load_data import Data
from parameter import Parameter
from fc_net import FCNet
from helper_gen import make_parser, performance, optimize

pwd = os.getcwd()
        
if __name__ == '__main__':

    args = make_parser()

    param_file = args.param

    params = Parameter(param_file, pwd)
    input_folder = args.input
    view = args.view
    result_file = args.result
    
    data = Data(input_folder, params.trainbs, params.testbs, view)
    train_loader, test_loader = data.loaders()
    test_set = list(test_loader)
    
    train_batchlists = list(train_loader)
    
    size = len(train_batchlists[0][0][0])
    fc_net = FCNet(size, view)
    
    optimize(fc_net, train_batchlists, params)

    perf_list = performance(fc_net, test_set)