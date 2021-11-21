import torch
import torch.nn as nn

## PLEASE replace this linear network with a convolutional NN according to the paper's architecture
## Make sure it outputs a binary value between 0 and 1, and takes in a variable number of inputs
## depending on whether param_global or param_local was chosen.


class Net_CNN(nn.Module):
    def __init__(self, view):
        super(Net_CNN, self).__init__()

        self.view = view

        self.local_length = 201
        self.global_length = 2001


        ## Convolutional Columns

        # Only local view use these layers
        self.conv_in_local = nn.Conv1d(1, 16, 5)  # (in channels, out channels, kernel size)
        self.conv_1_local = nn.Conv1d(16, 16, 5)
        self.conv_2_local = nn.Conv1d(16, 32, 5)
        self.conv_3_local = nn.Conv1d(32, 32, 5)

        # Only global view uses these layers
        self.conv_in_global = nn.Conv1d(1, 16, 5)
        self.conv_1_global = nn.Conv1d(16, 16, 5)
        self.conv_2_global = nn.Conv1d(16, 32, 5)
        self.conv_3_global = nn.Conv1d(32, 32, 5)
        self.conv_4_global = nn.Conv1d(32, 64, 5)
        self.conv_5_global = nn.Conv1d(64, 64, 5)
        self.conv_6_global = nn.Conv1d(64, 128, 5)
        self.conv_7_global = nn.Conv1d(128, 128, 5)
        self.conv_8_global = nn.Conv1d(128, 256, 5)
        self.conv_9_global = nn.Conv1d(256, 256, 5)


        ## Maxpool layers
        # Local view
        self.maxpool_local = nn.MaxPool1d(7, stride=2)

        # Global view
        self.maxpool_global = nn.MaxPool1d(5, stride=2)



        ## Fully connected layers
        # Local view
        self.fc_in_local = nn.Linear(32 * 5, 512, bias=True)  # output of conv1d is 32*5 ? --> only *5 once because 1d not 2d convolution?

        # Global view
        self.fc_in_global = nn.Linear(256 * 5, 512, bias=True)  # output of conv1d is 256*5 ? --> only *5 once because 1d not 2d convolution?

        # Both global and local view

        self.fc_in_both = nn.Linear(32*5 + 256*5, 512, bias=True)  # output of conv1d is 32*5 + 256*5 ? --> only *5 once because 1d not 2d convolution?

        ## Don't understand: First linear FC layer can't take in both local and global views the same way, since the outputs of the last convolutional
        ## layers in each "column" are of different sizes (32*5 for local, 256*5 for global)
        ## Are outputs of two convolutional columns concatenated (as in linear network)??

        self.fc_1 = nn.Linear(512, 512, bias=True)
        self.fc_2 = nn.Linear(512, 512 , bias=True)
        self.fc_3 = nn.Linear(512, 512, bias=True)
        self.fc_out = nn.Linear(512, 1, bias=True)

    # Feed forward
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


    # Feed Forward for convolutional column for local view
    def forward_conv_local(self, input):
        c1 = torch.relu(self.conv_in_local(input))
        c2 = torch.relu(self.conv_1_local(c1))
        m1 = self.maxpool_local(c2)             # Need relu? --> No
        c3 = torch.relu(self.conv_2_local(m1))
        c4 = torch.relu(self.conv_3_local(c3))
        m2 = self.maxpool_local(c4)           # Need relu? --> No
        
        return m2

    # Feed Forward for convolutional column for global view
    def forward_conv_global(self, input):
        c1 = torch.relu(self.conv_in_global(input))
        c2 = torch.relu(self.conv_1_global(c1))
        m1 = self.maxpool_global(c2)             # Need relu? --> No
        c3 = torch.relu(self.conv_2_global(m1))
        c4 = torch.relu(self.conv_3_global(c3))
        m2 = self.maxpool_global(c4)           # Need relu? --> No
        c5 = torch.relu(self.conv_4_global(m2))
        c6 = torch.relu(self.conv_5_global(c5))
        m3 = self.maxpool_global(c6)
        c7 = torch.relu(self.conv_6_global(m3))
        c8 = torch.relu(self.conv_7_global(c7))
        m4 = self.maxpool_global(c8)
        c9 = torch.relu(self.conv_8_global(m4))
        c10 = torch.relu(self.conv_9_global(c9))
        m5 = self.maxpool_global(c10)        
        
        return m5

    # Reset parameters if using net more than once
    def reset(self):
        self.conv_in_local.reset_parameters()
        self.conv_1_local.reset_parameters()
        self.conv_2_local.reset_parameters()
        self.conv_3_local.reset_parameters()

        self.conv_in_global.reset_parameters()
        self.conv_1_global.reset_parameters()
        self.conv_2_global.reset_parameters()
        self.conv_3_global.reset_parameters()
        self.conv_4_global.reset_parameters()
        self.conv_5_global.reset_parameters()
        self.conv_6_global.reset_parameters()
        self.conv_7_global.reset_parameters()
        self.conv_8_global.reset_parameters()
        self.conv_9_global.reset_parameters()

        self.fc_in_local.reset_parameters()
        self.fc_in_global.reset_parameters()
        self.fc_in_both.reset_parameters()
        self.fc_1.reset_parameters()
        self.fc_2.reset_parameters()
        self.fc_3.reset_parameters()
        self.fc_out.reset_parameters()



class CustomLoss(nn.Module):

    def __init__(self, data, loss_fn):
        super(CustomLoss, self).__init__()
        
        # The dataset probs don't change so they can be determined once
        self.data_probability(data)
        self.loss_fn = loss_fn

    def forward(self, model_prob, known_prob):
        '''
        loss function that determines the sum of differeneces from the model probability and the known probability
        model_prob : nd-array containing probability that exoplanet is detected from the model
        known_prob : nd-array containing 1s for autovetter known planet or 0 for not planet
        '''

        loss = sum(model_prob - known_prob)
        return loss