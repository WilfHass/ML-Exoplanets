import torch
class LinNet(torch.nn.Module):
    def __init__(self,size):
        super(LinNet, self).__init__() 
        # super gives access to attributes in a superclass from the subclass that inherits from it
        self.fc1= torch.nn.Linear(size, 1)

    def forward(self, x):
        b1 = torch.sigmoid(self.fc1(x))
        y = b1

#         y = self.fc1(x)
        return y