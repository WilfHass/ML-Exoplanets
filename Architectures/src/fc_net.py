import torch.nn as nn
import torch

class FCNet(nn.Module):
    def __init__(self, size, view):
        '''
        Fully Connected Neural Network
        size : size of data
        view : view of TCE data -> 'global' | 'local' | 'both'
        '''
        super(FCNet, self).__init__() # super gives access to attributes in a superclass from the subclass that inherits from it
        self.view = view
        
        if view == 'both':
            # Local view FC layers 
            self.a1 = nn.Linear(201,100)
            self.a2 = nn.Linear(100,50)
            self.a3 = nn.Linear(50,25)

            # Global view FC layers
            self.b1 = nn.Linear(2000,1000)
            self.b2 = nn.Linear(1000,500)
            self.b3 = nn.Linear(500,250)
            self.b4 = nn.Linear(250,100)
            self.b5 = nn.Linear(100,50)
            self.b6 = nn.Linear(50,25)

            # Combined layers
            self.c1 = nn.Linear(25,10)
            self.c2 = nn.Linear(10,1)

        else:
            # FC layers -- HELPPPP: Shouldn't we use the same architecture for a local view as the architecture in the 'both' views 
            self.fc1= nn.Linear(size, 150)
            self.fc2= nn.Linear(150, 100) 
            self.fc3= nn.Linear(100, 50)
            self.fc4= nn.Linear(50, 25)
            self.fc5= nn.Linear(25, 1)
            

    def forward(self, x):
        '''
        Feed forward propogation
        x : data input of batch size by 2201
        '''
        if self.view == "both":

            # DOUBLE CHECK SLICING (upper limit is exclusive so only goes up to 200 and 2200)
            x1 = x[:, :201]
            x2  = x[:, 201:2201]

            # Local View 
            d1 = torch.relu(self.a1(x1))
            d2 = torch.relu(self.a2(d1))
            d3 = torch.relu(self.a3(d2))
            
            # Global View
            e1 = torch.relu(self.b1(x2))
            e2 = torch.relu(self.b2(e1))
            e3 = torch.relu(self.b3(e2))
            e4 = torch.relu(self.b4(e3))
            e5 = torch.relu(self.b5(e4))
            e6 = torch.relu(self.b6(e5))
            
            # Combine Views
            g1 = torch.relu(self.c1(d3+e6))
            y = torch.sigmoid(self.c2(g1))

        else:
            b1 = torch.relu(self.fc1(x))
            b2 = torch.relu(self.fc2(b1))
            b3 = torch.relu(self.fc3(b2))
            b4 = torch.relu(self.fc4(b3))
            y = torch.sigmoid(self.fc5(b4))
            
        return y