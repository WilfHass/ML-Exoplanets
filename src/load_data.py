import torch
import os

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

pwd = os.getcwd()

def getData(input_folder):
    '''
    getData turns the pytorch tensor files into a concatenated tensor
    input_folder : path to input data folder
    '''
    train_data = torch.tensor([])
    test_data = torch.tensor([])

    for filename in os.listdir(pwd + "/" + input_folder):
        model = torch.load(pwd + "/" + input_folder + '/' + filename)

        if "test" in filename:
            test_data = torch.cat((test_data, model), 0)
        else:
            train_data = torch.cat((train_data, model), 0)

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
        data = self.data[idx, :2202]
        labels = list(self.data[idx, 2202:]) # Gets the label, kepid, tce_plnt_num

        return data, labels


def dataPrep(input_folder, batch_size, test_batch_size):
    '''Prepare the data by placing it into Datasets and then into the DataLoader from PyTorch'''
    # get the train and test data
    train_data, test_data = getData(input_folder)

    # Put data into the datasets
    train_dataset = Data(train_data)
    test_dataset = Data(test_data)

    # Load the data into the pyTorch DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   # shuffle the training data
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)    # don't shuffle the testing data

    return train_loader, test_loader