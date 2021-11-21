import torch
import torchvision
import os
import torchvision.transforms as transforms
import numpy as np
pwd = os.getcwd()
class Data():
    def __init__(self, input_folder,train_batchsize,test_batchsize):
        self.train_bs = train_batchsize
        self.test_bs = test_batchsize
        train_list = []
        test_list = []
        for filename in os.listdir(pwd+"/"+input_folder):
            if "train" in filename:
                model = torch.load(pwd+"/"+input_folder+'/'+filename)
                for i in model:
                    train_list.append(i.detach().numpy())
            else:
                model = torch.load(pwd+"/"+input_folder+'/'+filename)
                for i in model:
                    test_list.append(i.detach().numpy())
                    
        train_tensor = torch.tensor(train_list)
        test_tensor = torch.tensor(test_list)

        self.trainset = train_tensor
        self.testset = test_tensor
         
    def loaders(self):
        train_loader = torch.utils.data.DataLoader(self.trainset, batch_size = self.train_bs, shuffle = True, num_workers = 2)
        test_loader = torch.utils.data.DataLoader(self.testset, batch_size = self.test_bs, shuffle = False, num_workers = 2)
        return train_loader,test_loader