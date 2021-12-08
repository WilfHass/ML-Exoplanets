import torch

class LinNet(torch.nn.Module):
    def __init__(self, view, device):
        '''
        Linear Neural Network with one layer, no hidden layers
        view : view of TCE data -> 'global' | 'local' | 'both'
        device : 'cuda' or 'cpu'
        '''
        super(LinNet, self).__init__() 
        self.view = view
        self.device = device

        if view =='local':
            size = 201
        elif view == 'global':
            size = 2001
        elif view == 'both':
            size = 2202
            
        self.fc1= torch.nn.Linear(size, 1)

    def forward(self, x):
        '''
        x : data input of size (batch size, 2201)
        '''

        x = x.to(self.device)
        if self.view == 'local':
            x = x[:, :201]
        elif self.view == 'global':
            x = x[:, 201:]
            
        y = torch.sigmoid(self.fc1(x))
        return y


    def reset(self):
        '''
        Reset parameters if training net more than once
        '''
        self.fc1.reset_parameters()