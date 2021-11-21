import torch
class FCNet(torch.nn.Module):
    def __init__(self,size,view):
        super(FCNet, self).__init__() 
        self.view = view
        # super gives access to attributes in a superclass from the subclass that inherits from it
        self.fc1= torch.nn.Linear(size, 150)
        self.fc2= torch.nn.Linear(150, 100) 
        self.fc3= torch.nn.Linear(100, 50)
        self.fc4= torch.nn.Linear(50, 25)
        self.fc5= torch.nn.Linear(25, 1)
        
        
        self.a1 = torch.nn.Linear(201,100)
        self.a2 = torch.nn.Linear(100,50)
        self.a3 = torch.nn.Linear(50,25)
        
        self.b1 = torch.nn.Linear(2001,1000)
        self.b2 = torch.nn.Linear(1000,500)
        self.b3 = torch.nn.Linear(500,250)
        self.b4 = torch.nn.Linear(250,100)
        self.b5 = torch.nn.Linear(100,50)
        self.b6 = torch.nn.Linear(50,25)
        
        self.c1 = torch.nn.Linear(50,25)
        self.c2 = torch.nn.Linear(25,1)

    def forward(self, x):
        if self.view == "both":
            d1 = torch.nn.functional.relu(self.a1(x[0:201]))
            d2 = torch.nn.functional.relu(self.a2(d1))
            d3 = torch.nn.functional.relu(self.a3(d2))
            
            e1 = torch.nn.functional.relu(self.b1(x[201:2201]))
            e2 = torch.nn.functional.relu(self.b2(e1))
            e3 = torch.nn.functional.relu(self.b3(e2))
            e4 = torch.nn.functional.relu(self.b4(e3))
            e5 = torch.nn.functional.relu(self.b5(e4))
            e6 = torch.nn.functional.relu(self.b6(e5))
            
            g1 = torch.nn.functional.relu(self.c1(d3+e6))
            y = torch.sigmoid(self.c2(g1))
        else:
            b1 = torch.nn.functional.relu(self.fc1(x))
            b2 = torch.nn.functional.relu(self.fc2(b1))
            b3 = torch.nn.functional.relu(self.fc3(b2))
            b4 = torch.nn.functional.relu(self.fc4(b3))
            y = torch.sigmoid(self.fc5(b4))
        return y