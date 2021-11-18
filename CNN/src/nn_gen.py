import torch
import torch.nn as nn

## PLEASE replace this linear network with a convolutional NN according to the paper's architecture
## Make sure it outputs a binary value between 0 and 1, and takes in a variable number of inputs
## depending on whether param_global or param_local was chosen.

## Aviv: Can't do this, because the local and global CNNs have different numbers of convolutional layers,
## so two different networks will be needed

class Net_CNN(nn.Module):
    def __init__(self, view, per_layer_conv, per_layer_fc):
        super(Net_CNN, self).__init__()

        self.view = view

        self.local_length = 201
        self.global_length = 2001

        #self.conv_in = nn.Conv1d(201, 16, 5)        # Da fuck????
        #self.conv_1 = nn.Conv1d(16, 16, 5)          # Da fuck????
        #self.conv_2 = nn.Conv1d(32, 32, 5)          # Da fuck????
        #self.conv_3 = nn.Conv1d(32, 32, 5)          # Da fuck????

        # Both local and global view use these layers
        self.conv_in = nn.Conv1d(1, 16, 5)  # (in channels, out channels, kernel size)
        self.conv_1 = nn.Conv1d(16, 16, 5)
        self.conv_2 = nn.Conv1d(16, 32, 5)
        self.conv_3 = nn.Conv1d(32, 32, 5)

        # Only global view uses these layers
        self.conv_4 = nn.Conv1d(32, 64, 5)
        self.conv_5 = nn.Conv1d(64, 64, 5)
        self.conv_6 = nn.Conv1d(64, 128, 5)
        self.conv_7 = nn.Conv1d(128, 128, 5)
        self.conv_8 = nn.Conv1d(128, 256, 5)
        self.conv_9 = nn.Conv1d(256, 256, 5)

        # Local view
        self.maxpool_local = nn.MaxPool1d(7, stride=2)

        # Global view
        self.maxpool_global = nn.MaxPool1d(5, stride=2)

        #self.fc_in = nn.Linear(per_layer_fc[0], per_layer_fc[1], bias=True)
        #self.fc_1 = nn.Linear(per_layer_fc[1], per_layer_fc[2], bias=True)
        #self.fc_2 = nn.Linear(per_layer_fc[2], per_layer_fc[3], bias=True)
        #self.fc_3 = nn.Linear(per_layer_fc[3], per_layer_fc[4], bias=True)
        #self.fc_out = nn.Linear(per_layer_fc[4], per_layer_fc[5], bias=True)

        # Local view
        self.fc_in_local = nn.Linear(32 * 5, 512, bias=True)  # output of conv1d is 32*5 ? --> only *5 once because 1d not 2d convolution?

        # Global view
        self.fc_in_global = nn.Linear(256 * 5, 512, bias=True)  # output of conv1d is 256*5 ? --> only *5 once because 1d not 2d convolution?

        # Both global and local view
        self.fc_in_both = nn.Linear(32*5 + 256*5, 512, bias=True)  # output of conv1d is 256*5 ? --> only *5 once because 1d not 2d convolution?

        ## Don't understand: First linear FC layer can't take in both local and global views the same way, since the outputs of the last convolutional
        ## layers in each "column" are of different sizes (32*5 for local, 256*5 for global)
        ## Are outputs of two convolutional columns concatenated (as in linear network)??

        self.fc_1 = nn.Linear(512, 512, bias=True)
        self.fc_2 = nn.Linear(512, 512 , bias=True)
        self.fc_3 = nn.Linear(512, 512, bias=True)
        self.fc_out = nn.Linear(512, 1, bias=True)



    def forward(self, input_local, input_global):
        if (self.view == "local"):
            conv_output = self.forward_conv_local(input_local)
            h1 = torch.relu(self.fc_in_local(conv_output))

        elif (self.view == "global"):
            conv_output = self.forward_conv_global(input_global)
            h1 = torch.relu(self.fc_in_global(conv_output))

        elif (self.view == "both"):
            local_conv_output = self.forward_conv_local(input_local)
            global_conv_output = self.forward_conv_global(input_global)

            # Concatenate tensors from outputs of two different convolutional columns?
            ## What is format of local and global view tensors???
            conv_output = torch.cat((local_conv_output, global_conv_output), dim=1)

            h1 = torch.relu(self.fc_in_both(conv_output))


        h2 = torch.relu(self.fc_1(h1))
        h3 = torch.relu(self.fc_2(h2))
        h4 = torch.relu(self.fc_3(h3))
        y = torch.sigmoid(self.fc_out(h4))

        return y

    def forward_conv_local(self, input):
        c1 = torch.relu(self.conv_in(input))
        c2 = torch.relu(self.conv_1(c1))
        m1 = self.maxpool_local(c2)             # Need relu? --> No
        c3 = torch.relu(self.conv_2(m1))
        c4 = torch.relu(self.conv_3(c3))
        m2 = self.maxpool_local(c4)           # Need relu? --> No
        
        return m2

    def forward_conv_global(self, input):
        c1 = torch.relu(self.conv_in(input))
        c2 = torch.relu(self.conv_1(c1))
        m1 = self.maxpool_global(c2)             # Need relu? --> No
        c3 = torch.relu(self.conv_2(m1))
        c4 = torch.relu(self.conv_3(c3))
        m2 = self.maxpool_global(c4)           # Need relu? --> No
        c5 = torch.relu(self.conv_4(m2))
        c6 = torch.relu(self.conv_5(c5))
        m3 = self.maxpool_global(c6)
        c7 = torch.relu(self.conv_6(m3))
        c8 = torch.relu(self.conv_7(c7))
        m4 = self.maxpool_global(c8)
        c9 = torch.relu(self.conv_8(m4))
        c10 = torch.relu(self.conv_9(c9))
        m5 = self.maxpool_global(c10)        
        
        return m5

    def reset(self):
        self.conv_in.reset_parameters()
        self.conv_1.reset_parameters()
        self.conv_2.reset_parameters()
        self.conv_3.reset_parameters()
        self.conv_4.reset_parameters()
        self.conv_5.reset_parameters()
        self.conv_6.reset_parameters()
        self.conv_7.reset_parameters()
        self.conv_8.reset_parameters()
        self.conv_9.reset_parameters()

        #self.maxpool_1.reset_parameters()          # Maxpool doesn't need reset since it doesn't have any learnable weights?
        #self.maxpool_out.reset_parameters()
        #self.maxpool_local.reset_parameters()
        #self.maxpool_global.reset_parameters()

        self.fc_in_local.reset_parameters()
        self.fc_in_global.reset_parameters()
        self.fc_in_both.reset_parameters()
        self.fc_1.reset_parameters()
        self.fc_2.reset_parameters()
        self.fc_3.reset_parameters()
        self.fc_out.reset_parameters()