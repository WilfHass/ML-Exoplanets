import torch
class LinNet(torch.nn.Module):
    def __init__(self,view):
        super(LinNet, self).__init__() 
        # super gives access to attributes in a superclass from the subclass that inherits from it

        self.view = view
        if view =='local':
            size = 201
        elif view == 'global':
            size = 2001
        elif view == 'both':
            size = 2202
            
        self.fc1= torch.nn.Linear(size, 1)

    def forward(self, x):
        if self.view == 'local':
            x = x[:, :201]
        elif self.view == 'global':
            x = x[:, 201:]
            
        b1 = torch.sigmoid(self.fc1(x))
        y = b1

        # y = self.fc1(x)
        return y