import torch
import torch.nn as nn

## PLEASE replace this linear network with a convolutional NN according to the paper's architecture
## Make sure it outputs a binary value between 0 and 1, and takes in a variable number of inputs
## depending on whether param_global or param_local was chosen.

class Net(nn.Module):
    def __init__(self, per_layer):
        super(Net, self).__init__()
        self.fc_in = nn.Linear(per_layer[0], per_layer[1], bias=True)
        self.fc_1 = nn.Linear(per_layer[1], per_layer[2], bias=True)
        self.fc_2 = nn.Linear(per_layer[2], per_layer[3], bias=True)
        self.fc_3 = nn.Linear(per_layer[3], per_layer[4], bias=True)
        self.fc_out = nn.Linear(per_layer[4], per_layer[5], bias=True)

    def forward(self, h):
        h1 = torch.tanh(self.fc_in(h))
        h2 = torch.tanh(self.fc_1(h1))
        h3 = torch.tanh(self.fc_2(h2))
        h4 = torch.tanh(self.fc_3(h3))
        y = torch.tanh(self.fc_out(h4))
        return y

    def reset(self):
        self.fc_in.reset_parameters()
        self.fc_1.reset_parameters()
        self.fc_2.reset_parameters()
        self.fc_3.reset_parameters()
        self.fc_out.reset_parameters()