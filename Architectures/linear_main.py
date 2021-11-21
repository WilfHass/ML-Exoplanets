import sys, os
import numpy as np
import json, argparse, sys
import csv
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.append('src') 
from load_data import Data
from parameter import Parameter
# from linear_net import LinNet
pwd = os.getcwd()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Architecture')

    parser.add_argument('--view', type=str, help="Input 'local','global', or 'both'")
    parser.add_argument('--param',type=str,default='param/linear_params.json',help='location of params file')
    parser.add_argument('--input',type=str,default='torch_data',help='location of folder for data')
    args = parser.parse_args()

    param_file = args.param
    params = Parameter(param_file,pwd)
    input_folder = args.input
    view = args.view
    
    data = Data(input_folder,params.trainbs,params.testbs)
    train_data,test_data = data.loaders()
    
    lin_net = LinNet(view)
    