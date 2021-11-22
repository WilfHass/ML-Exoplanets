import sys, os
import numpy as np
import sys
import torch

sys.path.append('src') 
from load_data import Data
from parameter import Parameter
from linear_net import LinNet
from helper_gen import make_parser, performance, optimize

pwd = os.getcwd()


if __name__ == '__main__':

    args = make_parser()

    param_file = args.param
    params = Parameter(param_file,pwd)
    input_folder = args.input
    view = args.view
    result_file = args.result
    
    data = Data(input_folder, params.trainbs, params.testbs, view)
    train_loader, test_loader = data.loaders()
    test_set = list(test_loader)
    
    train_batchlists = list(train_loader)
    
    size = len(train_batchlists[0][0][0])
    lin_net = LinNet(size)
    
    optimize(lin_net,train_batchlists,params)
    performance(lin_net, test_set)