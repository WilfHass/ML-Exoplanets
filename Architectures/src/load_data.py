import torch
import torchvision
import os
import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

pwd = os.getcwd()

def getData(input_folder):

    train_data = torch.tensor([])
    test_data = torch.tensor([])

    for filename in os.listdir(pwd + "/" + input_folder):
        model = torch.load(pwd + "/" + input_folder + '/' + filename)

        if "train" in filename:
            train_data = torch.cat((train_data, model), 0)

        else:
            test_data = torch.cat((test_data, model), 0)

    return train_data, test_data


class Data(Dataset):
    ''' Creates train or test dataset with methods to get individual samples'''

    def __init__(self, data):
        self.data = data

    def __len__(self):
        ''' Gets the length of the dataset'''
        return len(self.data)

    def __getitem__(self, idx):
        ''' Gets an individual sample given an index'''
        data = self.data[idx, :-1]
        label = self.data[idx, -1] # Gets the label of the img
        return data, label


def dataPrep(input_folder, batch_size):
    '''Prepare the data by placing it into Datasets and then into the DataLoader from PyTorch'''
    # get the train and test data
    train_data, test_data = getData(input_folder)

    # Put data into the datasets
    train_dataset = Data(train_data)
    test_dataset = Data(test_data)

    # Load the data into the pyTorch DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   # shuffle the training data
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    # don't shuffle the testing data

    return train_loader, test_loader


# class Data():

#     def __init__(self, input_folder, train_batchsize, test_batchsize, view):

#         self.train_bs = train_batchsize
#         self.test_bs = test_batchsize
#         train_list = []
#         test_list = []

#         for filename in os.listdir(pwd + "/" + input_folder):
#             model = torch.load(pwd + "/" + input_folder + '/' + filename)

#             if "train" in filename:
#                 for i in model:
#                     train_list.append(list(i.detach().numpy()))
#             else:
#                 for i in model:
#                     test_list.append(list(i.detach().numpy()))
        
#         train_inputs = []
#         train_target = []

#         for i in train_list:

#             if view == 'local':
#                 new_line = i[0:201]
#                 train_target.append(i[-1])
#                 train_inputs.append(new_line)

#             elif view=='global':
#                 new_line = i[201:2202]
#                 train_target.append(i[-1])
#                 train_inputs.append(new_line)

#             else:
#                 new_line = i[0:2202]
#                 train_target.append(i[-1])
#                 train_inputs.append(new_line)
                
#         train_set =[]

#         for i in range(len(train_target)):
#             pair = (torch.tensor(train_inputs[i]),train_target[i])
#             train_set.append(pair)
                    
#         test_inputs = []
#         test_target = []

#         for i in test_list:
#             if view == 'local':
#                 new_line = i[0:201]
#                 test_target.append(i[-1])
#                 test_inputs.append(new_line)

#             elif view=='global':
#                 new_line = i[201:2202]
#                 test_target.append(i[-1])
#                 test_inputs.append(new_line)

#             else:
#                 new_line = i[0:2202]
#                 test_target.append(i[-1])
#                 test_inputs.append(new_line)
                
#         test_set =[]

#         for i in range(len(test_target)):
#             pair = (torch.tensor(test_inputs[i]), test_target[i])
#             test_set.append(pair)

#         self.train_set = train_set
#         self.test_set = test_set
         
    # def loaders(self):
    #     train_loader = torch.utils.data.DataLoader(self.train_set, batch_size = self.train_bs, shuffle = True, num_workers = 2)
    #     test_loader = torch.utils.data.DataLoader(self.test_set, batch_size = self.test_bs, shuffle = False, num_workers = 2)
    #     return train_loader, test_loader